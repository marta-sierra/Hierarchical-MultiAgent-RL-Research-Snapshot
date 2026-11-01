"""Run one recorded episode for the dashboard: traffic -> Simulation -> policies, plus a full trace.

The simulator is not modified: the recorder wraps the methods of its components (switch,
servers, executors) on this one instance and logs every mechanical event with `sim.now_ms`.
Everything the browser shows comes from that log, from the decision requests the loop sees,
and from `sim.finalize()`.
"""
from __future__ import annotations

import functools
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import acmad.policies  # noqa: F401  (registers the policies)
from acmad.interface import ServerObs, SwitchObs, available, make
from acmad.interface.observations import FEATURE_NAMES
from acmad.sim import SimConfig, Simulation, Topology
from acmad.sim.links import SUMMARY_FIELDS
from acmad.sim.models import TaskFeatures, matlab_cpu_processing
from acmad.sim.models.cloudric_ldpc import CloudricLdpcModel, load_tables
from acmad.traffic import TaskStream, generate, load_config
from acmad.traffic.config import LOAD_CLASSES, SPLITS

REPO = Path(__file__).resolve().parents[3]
TOPOLOGY_DIR = REPO / "configs" / "topology"
SIM_DEFAULT = REPO / "configs" / "sim" / "default.yaml"


# ---------------------------------------------------------------------- parameters
@dataclass
class EpisodeParams:
    seed: int = 0
    split: str = "train"
    load_class: str = "any"            # any | low | medium | high (ADR-0007)
    n_dus: int = 3
    s: float = 4.0
    horizon_ms: float = 100.0          # keep jobs with t_arrival_ms < horizon_ms
    topology: str = "example"
    deadline_ms: float = 1.0
    downlink_ms: float = 0.0
    uplink_ms: float = 0.0
    estimator: str = "last_report"
    switch_policy: str = "round_robin"
    server_policy: str = "round_robin"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeParams":
        p = cls()
        for k, v in d.items():
            if not hasattr(p, k):
                raise ValueError(f"unknown parameter {k!r}")
            setattr(p, k, type(getattr(p, k))(v))
        if p.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if p.load_class != "any" and p.load_class not in LOAD_CLASSES:
            raise ValueError(f"load_class must be 'any' or one of {LOAD_CLASSES}")
        if not p.horizon_ms > 0:
            raise ValueError("horizon_ms must be > 0")
        return p


def topologies() -> dict[str, dict]:
    """Every `configs/topology/*.yaml`, by file stem."""
    return {f.stem: yaml.safe_load(f.read_text()) for f in sorted(TOPOLOGY_DIR.glob("*.yaml"))}


def options() -> dict[str, Any]:
    """What the setup form offers, with defaults."""
    sim = yaml.safe_load(SIM_DEFAULT.read_text())
    d = EpisodeParams(deadline_ms=sim["deadline_ms"], downlink_ms=sim["links"]["downlink_ms"],
                      uplink_ms=sim["links"]["uplink_ms"], estimator=sim["estimator"])
    return {"defaults": asdict(d), "splits": list(SPLITS),
            "load_classes": ["any", *LOAD_CLASSES], "topologies": topologies(),
            "policies": {"switch": list(available("switch")), "server": list(available("server"))},
            "estimators": ["none", "last_report"]}


def truncate(stream: TaskStream, horizon_ms: float) -> TaskStream:
    jobs = stream.jobs[stream.jobs.t_arrival_ms < horizon_ms].reset_index(drop=True)
    tasks = stream.tasks[stream.tasks.job_id.isin(jobs.job_id)].reset_index(drop=True)
    return TaskStream(jobs, tasks, {**stream.meta, "horizon_ms": horizon_ms})


def build_topology(cfg: dict) -> Topology:
    """The chosen topology; the dashboard explains only the shipped model of each kind."""
    for s in cfg["servers"]:
        for e in s["executors"]:
            if e.get("model") not in (None, e["kind"]):
                raise ValueError(f"dashboard supports only the default model per kind, got {e['model']!r}")
    return Topology.from_dict(cfg)


# ---------------------------------------------------------------------- service-model explanation
def explain_model(ex, task_id: int, f: TaskFeatures) -> dict[str, Any]:
    """Re-run the executor's model on the task's keyed rng, step by step (same draw, ADR-0002)."""
    real, spec = ex.model, ex.spec
    if f.kind == "cpu_processing":
        m = matlab_cpu_processing.load()
        a, b, c, d = m.coef
        g = f.nprb * matlab_cpu_processing.RE_PER_PRB * f.qm
        t0 = m.median_ms(f.nprb, g, f.total_bits)
        idx = int(ex.rng(task_id).integers(len(m.residuals)))
        eps = float(m.residuals[idx])
        return {"model": "MATLAB cpu_processing fit / k (ADR-0008)", "steps": [
            ["features (nPRB, G, total_bits)", f"({f.nprb}, {g}, {f.total_bits})"],
            ["t_MATLAB = a + b·nPRB + c·G + d·total_bits [ms]", a + b * f.nprb + c * g + d * f.total_bits],
            ["a [ms] kept?", f"{a:.4f} {'yes' if m.keep_constant else 'no (dropped)'}"],
            ["k", m.k],
            ["median time t0 [ms]", t0],
            ["residual pool size / index", f"{len(m.residuals)} / {idx}"],
            ["eps_t", f"{eps:+.4f}"],
            ["t = t0 * exp(eps_t) [ms]", t0 * float(np.exp(eps))], ["E [J] (not modelled)", 0.0]],
            "service_time_ms": t0 * float(np.exp(eps)), "dynamic_energy_j": 0.0}
    if not isinstance(real, CloudricLdpcModel):
        s, e = real(f, spec, ex.rng(task_id))
        return {"model": type(real).__name__, "steps": [], "service_time_ms": s, "dynamic_energy_j": e}
    tab = load_tables()[real.KIND]
    rng = ex.rng(task_id)
    t0, e0 = float(tab.t_ms[f.mcs, f.nprb - 1, f.snr_db]), float(tab.e_j[f.mcs, f.nprb - 1, f.snr_db])
    band = int(np.searchsorted(tab.band_edges, f.total_bits, side="right"))
    pool = tab.res[band]
    idx = int(rng.integers(len(pool)))
    eps_t, eps_e = (float(x) for x in pool[idx])
    t = t0 * float(np.exp(eps_t))
    e_tot = e0 * float(np.exp(eps_e))
    e_dyn = max(e_tot - tab.static_power_w * t * 1e-3, 0.0)
    edges = [int(x) for x in tab.band_edges]
    return {"model": f"CloudRIC ldpc lookup ({real.KIND})", "steps": [
        ["key (MCS, nPRB, SNR dB)", f"({f.mcs}, {f.nprb}, {f.snr_db})"],
        ["median time t0 [ms]", t0], ["median energy E0 [J]", e0],
        ["total_bits -> band", f"{f.total_bits} -> {band}  (edges {edges})"],
        ["residual pool size / index", f"{len(pool)} / {idx}"],
        ["eps_t, eps_E", f"{eps_t:+.4f}, {eps_e:+.4f}"],
        ["t = t0 * exp(eps_t) [ms]", t], ["E = E0 * exp(eps_E) [J]", e_tot],
        ["P_static [W]", tab.static_power_w], ["E_dyn = max(E - P_static*t, 0) [J]", e_dyn]],
        "service_time_ms": t, "dynamic_energy_j": e_dyn}


# ---------------------------------------------------------------------- recorder
class Recorder:
    """Logs mechanical events by wrapping methods of one Simulation instance and its components.

    Heap events (`arrival`, `deliver`, `end`, `deadline`) and `drop` wrap the engine's own
    handlers; the rest wrap the switch, servers and executors. `ctx` names the job / task of
    the decision being applied (the switch's `dispatch` does not get the job id).
    """

    HEAP = ("arrival", "deliver", "end", "deadline")

    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.events: list[dict] = []
        self.model_calls: dict[int, dict] = {}          # task_id -> explanation + recorded cost
        self.enqueue_seq: dict[int, int] = {}
        self.ctx: dict = {}
        self._seq = 0
        for kind in self.HEAP:
            key = "job" if kind in ("arrival", "deliver", "deadline") else "task"
            self._wrap(sim, f"_on_{kind}", lambda x, kind=kind, key=key: self.log(kind, **{key: int(x)}))
        self._wrap(sim, "_drop", lambda t, energy, outcome: self.log(
            "drop", task=t.id, outcome=outcome, status_before=t.status, energy_j=energy))
        sw = sim.switch
        self._wrap(sw, "dispatch", lambda server, now_ms, bits: self.log("dispatch", server=server, **self.ctx))
        self._wrap(sw, "on_ack", lambda server, summary, at: self.log(
            "ack", server=server, reported_at_ms=at, summary=np.asarray(summary).tolist()))
        for srv in sim.servers:
            self._wrap(srv, "receive", lambda ids, preds, k=srv.id: self.log("receive", server=k, tasks=list(ids)),
                       after=lambda out, ids, preds, k=srv.id: out and self.log("ready", server=k, tasks=list(out)))
            self._wrap(srv, "complete", None,
                       after=lambda out, tid, k=srv.id: out and self.log("release", server=k, tasks=list(out)))
            for i, ex in enumerate(srv.executors):
                self._wrap_executor(srv.id, i, ex)

    def stages(self) -> dict[int, dict[str, int]]:
        """Per task, the index in `events` where it entered each stage (absent = never)."""
        jobs = {j.id: j.tasks for j in self.sim.jobs.values()}
        st: dict[int, dict[str, int]] = {t: {} for t in self.sim.tasks}
        for ev in self.events:
            i, ty = ev["seq"], ev["type"]
            if ty in ("arrival", "dispatch", "receive_job"):
                for t in jobs[ev["job"]]:
                    st[t].setdefault(ty, i)
            elif ty in ("receive", "ready", "release"):
                for t in ev["tasks"]:
                    st[t].setdefault("ready" if ty == "release" else ty, i)
            elif ty in ("enqueue", "start"):
                st[ev["task"]][ty] = i
            elif ty == "decision_server":
                st[ev["task"]]["decision"] = i
            elif ty in ("finish", "drop"):
                st[ev["task"]].setdefault("end", i)
        return st

    def log(self, kind: str, **kw) -> None:
        self.events.append({"t": self.sim.now_ms, "seq": len(self.events), "type": kind, **kw})

    @staticmethod
    def _wrap(obj, name: str, before, after=None) -> None:
        orig = getattr(obj, name)

        @functools.wraps(orig)
        def w(*a, **kw):
            if before is not None:
                before(*a, **kw)
            out = orig(*a, **kw)
            if after is not None:
                after(out, *a, **kw)
            return out

        setattr(obj, name, w)

    def _wrap_executor(self, k: int, i: int, ex) -> None:
        def after_enqueue(out, tid, features, now_ms):
            s, e = ex._cost[tid]
            self.enqueue_seq[tid] = self._seq
            self._seq += 1
            info = explain_model(ex, tid, features)
            info |= {"server": k, "executor": i, "recorded_service_time_ms": s, "recorded_dynamic_energy_j": e,
                     "match": bool(np.isclose(info["service_time_ms"], s) and np.isclose(info["dynamic_energy_j"], e))}
            self.model_calls[tid] = info
            self.log("enqueue", server=k, executor=i, task=tid, service_time_ms=s)

        self._wrap(ex, "enqueue", None, after=after_enqueue)
        self._wrap(ex, "start_next", None, after=lambda out, now: [
            self.log("start", server=k, executor=i, task=t, end_ms=end) for t, end in out])
        self._wrap(ex, "finish", None, after=lambda out, tid: self.log(
            "finish", server=k, executor=i, task=tid, energy_j=out))
        self._wrap(ex, "remove_waiting", lambda tid: self.log("drop_waiting", server=k, executor=i, task=tid))
        self._wrap(ex, "abort", None, after=lambda out, tid, now: self.log(
            "abort", server=k, executor=i, task=tid, energy_j=out))


def _obs_json(obs: SwitchObs | ServerObs) -> dict:
    if isinstance(obs, SwitchObs):
        return {"now_ms": obs.now_ms, "job": obs.job.id, "slack_ms": obs.job.slack_ms,
                "cluster": {"arrival_rate_per_ms": obs.cluster.arrival_rate_per_ms, "mean_tbs": obs.cluster.mean_tbs,
                            "jobs_since_ack": obs.cluster.jobs_since_ack.tolist(),
                            "bits_since_ack": obs.cluster.bits_since_ack.tolist()},
                "reports": [{"summary": r.summary.tolist(), "reported_at_ms": r.reported_at_ms, "age_ms": r.age_ms}
                            for r in obs.reports],
                "estimates": None if obs.estimates is None else [np.asarray(e).tolist() for e in obs.estimates]}
    return {"now_ms": obs.now_ms, "server": obs.server_id, "task": obs.task.id, "slack_ms": obs.task.slack_ms,
            "message": None if obs.message.is_null else obs.message.vector.tolist(),
            "executors": [{"kind": e.kind, "capacity": e.capacity, "n_busy": e.n_busy, "n_queued": e.n_queued,
                           "elapsed_ms": e.elapsed_ms, "queued_bits": e.queued_bits, "allowed": e.allowed}
                          for e in obs.executors],
            "oracle": None if obs.oracle_cluster is None else obs.oracle_cluster.tolist()}


# ---------------------------------------------------------------------- run
def run_episode(raw: dict[str, Any]) -> dict[str, Any]:
    """Generate traffic, simulate with the chosen policies, return everything the page shows."""
    p = EpisodeParams.from_dict(raw)
    t0 = time.perf_counter()
    full = generate(load_config(), split=p.split, n_dus=p.n_dus, s=p.s, seed=p.seed,
                    load_class=None if p.load_class == "any" else p.load_class)
    stream = truncate(full, p.horizon_ms)
    topo_cfg = topologies()[p.topology]
    topo = build_topology(topo_cfg)
    config = SimConfig.from_dict({"deadline_ms": p.deadline_ms, "estimator": p.estimator,
                                  "links": {"downlink_ms": p.downlink_ms, "uplink_ms": p.uplink_ms}})
    sim = Simulation(stream, topo, config, seed=p.seed)
    rec = Recorder(sim)
    switch, server = make("switch", p.switch_policy), make("server", p.server_policy)
    switch.reset(p.seed)
    server.reset(p.seed + 1)       # TODO: seed derivation by name belongs to the experiment runner (ADR-0002)
    decisions = []
    while (req := sim.next_decision()) is not None:
        action = (switch if req.tier == "switch" else server).act(req.obs)
        d = {"id": req.decision_id, "t": sim.now_ms, "tier": req.tier, "obs": _obs_json(req.obs)}
        if req.tier == "switch":
            d["action"] = {"server": int(action.server),
                           "message": None if action.message.is_null else np.asarray(action.message.vector).tolist()}
        else:
            d["action"] = {"executor": int(action.executor)}
        decisions.append(d)
        if req.tier == "switch":
            rec.ctx = {"job": req.obs.job.id}
            rec.log("decision_switch", decision=req.decision_id, job=req.obs.job.id, server=int(action.server))
        else:
            rec.ctx = {"task": req.obs.task.id}
            rec.log("decision_server", decision=req.decision_id, task=req.obs.task.id, server=req.server,
                    executor=int(action.executor))
        sim.apply(action)
        rec.ctx = {}
    final = sim.finalize()
    records = final.to_frame()

    stages = rec.stages()
    tasks_df = stream.tasks.set_index("task_id")
    rec_by_task = records.set_index("task_id")
    tasks = []
    for tid, r in rec_by_task.iterrows():
        tr = tasks_df.loc[tid]
        tasks.append({
            "id": int(tid), "job": int(r.job_id), "kind": r.kind, "allowed": str(tr.allowed),
            "depends_on": None if pd.isna(tr.depends_on) else int(tr.depends_on),
            "features": {n: (float(tr[n]) if n == "code_rate" else int(tr[n])) for n in FEATURE_NAMES},
            "server": _int(r.server), "executor": _int(r.executor),
            "t_arrival_ms": r.t_arrival_ms,
            "t_deliver_ms": None if _f(r.server) is None else r.t_arrival_ms + p.downlink_ms,
            "t_ready_ms": _f(r.t_ready_ms), "t_start_ms": _f(r.t_start_ms), "t_end_ms": r.t_end_ms,
            "deadline_ms": r.deadline_ms, "outcome": r.outcome, "energy_dynamic_j": r.energy_dynamic_j,
            "on_period": _int(r.on_period),
            "enqueue_seq": rec.enqueue_seq.get(int(tid)), "stage": stages[int(tid)],
            "switch_decision": _int(r.switch_decision_id), "server_decision": _int(r.server_decision_id),
        })
    jobs = [{"id": int(j.job_id), "du": int(j.du), "t_arrival_ms": float(j.t_arrival_ms)}
            for j in stream.jobs.itertuples(index=False)]
    meta = {k: v for k, v in stream.meta.items() if not isinstance(v, (dict, list))}
    payload = {
        "params": asdict(p),
        "meta": meta, "full_stream_jobs": len(full.jobs),
        "topology": [[{"kind": e.kind, "capacity": e.capacity} for e in s] for s in topo.servers],
        "summary_fields": list(SUMMARY_FIELDS), "feature_names": list(FEATURE_NAMES),
        "jobs": jobs, "tasks": tasks, "events": rec.events, "decisions": decisions,
        "model_calls": {str(k): v for k, v in rec.model_calls.items()},
        "end_ms": max(sim.now_ms, max((t["t_end_ms"] for t in tasks), default=0.0)),
        "outcomes": records.outcome.value_counts().to_dict(),
        "on_periods": final.on_periods_frame().to_dict("records"),     # ADR-0004
        "elapsed_s": time.perf_counter() - t0,
    }
    payload["checks"] = checks(payload)
    return payload


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _int(x) -> int | None:
    v = _f(x)
    return None if v is None else int(v)


# ---------------------------------------------------------------------- invariants
def checks(ep: dict) -> list[dict]:
    """Invariants of the mechanics, verified on the recorded episode."""
    tasks = {t["id"]: t for t in ep["tasks"]}
    D = ep["params"]["deadline_ms"]
    out = []

    def add(name: str, bad: list, detail: str) -> None:
        out.append({"name": name, "ok": not bad, "detail": detail, "violations": bad[:20]})

    add("conservation: every task has an outcome",
        [t["id"] for t in tasks.values() if t["outcome"] not in ("met", "missed", "dropped")],
        f"{len(tasks)} tasks")
    add("met => t_end <= deadline", [t["id"] for t in tasks.values()
                                     if t["outcome"] == "met" and t["t_end_ms"] > t["deadline_ms"] + 1e-9], "")
    add("deadline = arrival + D", [t["id"] for t in tasks.values()
                                   if abs(t["deadline_ms"] - t["t_arrival_ms"] - D) > 1e-9], f"D = {D} ms")
    add("nothing ends after its deadline", [t["id"] for t in tasks.values()
                                           if t["t_end_ms"] > t["deadline_ms"] + 1e-9], "drops/aborts at D (ADR-0013)")
    add("dependency: ldpc starts after its cpu_processing ends",
        [t["id"] for t in tasks.values() if t["depends_on"] is not None and t["t_start_ms"] is not None
         and (tasks[t["depends_on"]]["outcome"] != "met"
              or t["t_start_ms"] < tasks[t["depends_on"]]["t_end_ms"] - 1e-9)], "")
    add("cpu_processing runs on the CPU (rule, ADR-0011)",
        [t["id"] for t in tasks.values() if t["kind"] == "cpu_processing" and t["executor"] is not None
         and ep["topology"][t["server"]][t["executor"]]["kind"] != "cpu"], "")
    add("server-policy decisions only for ldpc (rule bypass)",
        [t["id"] for t in tasks.values() if t["kind"] == "cpu_processing" and t["server_decision"] is not None], "")
    add("met => full service time (t_end - t_start = sigma)",
        [t["id"] for t in tasks.values() if t["outcome"] == "met" and not np.isclose(
            t["t_end_ms"] - t["t_start_ms"], ep["model_calls"][str(t["id"])]["recorded_service_time_ms"])], "")
    add("service model re-run = recorded draw (keyed rng, ADR-0002)",
        [int(k) for k, v in ep["model_calls"].items() if not v["match"]], f"{len(ep['model_calls'])} calls")

    # capacity and FIFO per executor, replaying start/finish/abort events
    busy: dict[tuple[int, int], int] = {}
    cap_bad, fifo_bad = [], []
    last_start_seq: dict[tuple[int, int], int] = {}
    for ev in ep["events"]:
        key = (ev.get("server"), ev.get("executor"))
        if ev["type"] == "start":
            busy[key] = busy.get(key, 0) + 1
            if busy[key] > ep["topology"][key[0]][key[1]]["capacity"]:
                cap_bad.append(ev["task"])
            s = tasks[ev["task"]]["enqueue_seq"]
            if s < last_start_seq.get(key, -1):
                fifo_bad.append(ev["task"])
            last_start_seq[key] = s
        elif ev["type"] in ("finish", "abort"):
            busy[key] -= 1
    add("executor capacity never exceeded", cap_bad, "replay of start/finish/abort")
    add("FIFO: tasks start in enqueue order per executor", fifo_bad, "")
    return out
