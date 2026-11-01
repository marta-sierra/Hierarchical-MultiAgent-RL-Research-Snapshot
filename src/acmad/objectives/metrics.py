"""Episode metrics from the per-task records (the ground truth, ARCHITECTURE §7).

Reported only, no ranking (author, 2026-10-22). A job meets its deadline when every task of its chain
is `met`; `missed` and `dropped` both count as misses (ADR-0013). Energy = dynamic (per task) +
static (per executor on-period, ADR-0004).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from acmad.sim.topology import Topology

LATENCY_QUANTILES = (50, 95, 99)


def _quantiles(x: np.ndarray, prefix: str) -> dict[str, float]:
    if len(x) == 0:
        return {f"{prefix}_mean": float("nan"), **{f"{prefix}_p{q}": float("nan") for q in LATENCY_QUANTILES},
                f"{prefix}_max": float("nan")}
    return {f"{prefix}_mean": float(x.mean()), **{f"{prefix}_p{q}": float(np.percentile(x, q)) for q in LATENCY_QUANTILES},
            f"{prefix}_max": float(x.max())}


def episode_metrics(tasks: pd.DataFrame, on_periods: pd.DataFrame, topology: Topology, T_ms: float) -> dict[str, Any]:
    """Scalars at the top level; per-server and per-executor lists under `servers` / `executors`.

    - `job_hit_rate` (h): share of jobs whose every task is met; `task_{met,missed,dropped}_rate`.
    - `latency_ms_*`: arrival -> end of the last task, over met jobs; `ldpc_wait_ms_*`: ready ->
      start of service of the ldpc tasks that started.
    - `energy_*_j`: dynamic, static, total over the episode; `energy_per_job_mj`; `power_mean_w`.
    - `ldpc_on_gpu_share`: share of the started ldpc tasks that ran on a GPU.
    - per executor: tasks started, `utilisation` = busy unit-time / (capacity * T), `on_fraction` =
      time switched on / T, energies; per server: job share, utilisation over all its units.
    """
    T_s = T_ms / 1e3
    g = tasks.groupby("job_id", sort=True)
    jobs = pd.DataFrame({"met": (tasks.outcome == "met").groupby(tasks.job_id).all(),
                         "arrival": g.t_arrival_ms.first(), "end": g.t_end_ms.max(), "server": g.server.first()})
    n_jobs, n_tasks = len(jobs), len(tasks)
    started = tasks[tasks.t_start_ms.notna()]
    ldpc = started[started.kind == "ldpc_decoding"]
    gpus = {(k, e) for k, srv in enumerate(topology.servers) for e, spec in enumerate(srv) if spec.kind == "gpu"}
    on_gpu = pd.MultiIndex.from_arrays([ldpc.server.astype(int), ldpc.executor.astype(int)]).isin(list(gpus))

    e_dyn = float(tasks.energy_dynamic_j.sum())
    e_static = float(on_periods.static_j.sum()) if len(on_periods) else 0.0
    out: dict[str, Any] = {
        "T_ms": float(T_ms), "n_jobs": n_jobs, "n_tasks": n_tasks,
        "job_hit_rate": float(jobs.met.mean()) if n_jobs else float("nan"),
        **{f"task_{o}_rate": float((tasks.outcome == o).mean()) for o in ("met", "missed", "dropped")},
        **_quantiles((jobs.end - jobs.arrival)[jobs.met].to_numpy(float), "latency_ms"),
        **_quantiles((ldpc.t_start_ms - ldpc.t_ready_ms).to_numpy(float), "ldpc_wait_ms"),
        "energy_dynamic_j": e_dyn, "energy_static_j": e_static, "energy_total_j": e_dyn + e_static,
        "energy_per_job_mj": (e_dyn + e_static) / n_jobs * 1e3 if n_jobs else float("nan"),
        "power_mean_w": (e_dyn + e_static) / T_s,
        "ldpc_on_gpu_share": float(on_gpu.mean()) if len(ldpc) else float("nan"),
    }

    busy = (started.t_end_ms - started.t_start_ms).groupby([started.server, started.executor]).sum()
    n_started = started.groupby(["server", "executor"]).size()
    dyn = tasks[tasks.executor.notna()].groupby(["server", "executor"]).energy_dynamic_j.sum()
    if len(on_periods):
        on_time = (on_periods.t_off_ms - on_periods.t_on_ms).groupby([on_periods.server, on_periods.executor]).sum()
        stat = on_periods.groupby(["server", "executor"]).static_j.sum()
    else:
        on_time = stat = pd.Series(dtype=float)
    execs = []
    for k, srv in enumerate(topology.servers):
        for e, spec in enumerate(srv):
            key = (k, e)
            execs.append({"server": k, "executor": e, "kind": spec.kind, "capacity": spec.capacity,
                          "n_tasks": int(n_started.get(key, 0)),
                          "utilisation": float(busy.get(key, 0.0)) / (spec.capacity * T_ms),
                          "on_fraction": float(on_time.get(key, 0.0)) / T_ms,
                          "energy_dynamic_j": float(dyn.get(key, 0.0)), "energy_static_j": float(stat.get(key, 0.0))})
    out["executors"] = execs
    job_share = jobs.server.value_counts(normalize=True)
    out["servers"] = [{"server": k, "job_share": float(job_share.get(k, 0.0)),
                       "utilisation": sum(x["utilisation"] * x["capacity"] for x in execs if x["server"] == k)
                       / sum(s.capacity for s in srv)}
                      for k, srv in enumerate(topology.servers)]
    return out


def flat(metrics: dict[str, Any]) -> dict[str, Any]:
    """One flat row: scalars, plus `s{k}_<field>` per server and `s{k}e{e}_<field>` per executor."""
    row = {k: v for k, v in metrics.items() if k not in ("servers", "executors")}
    for s in metrics.get("servers", []):
        row |= {f"s{s['server']}_{f}": v for f, v in s.items() if f != "server"}
    for x in metrics.get("executors", []):
        row |= {f"s{x['server']}e{x['executor']}_{f}": v for f, v in x.items()
                if f not in ("server", "executor", "kind", "capacity")}
    return row
