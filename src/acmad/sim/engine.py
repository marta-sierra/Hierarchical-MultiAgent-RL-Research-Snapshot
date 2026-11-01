"""Event-driven simulation engine (ADR-0012): own heap, float ms clock, caller-driven loop.

    sim = Simulation(stream, topology, SimConfig(), channel=..., estimator=...)
    while (req := sim.next_decision()) is not None:
        sim.apply(policy_for(req).act(req.obs))
    records = sim.finalize()

Events are ordered by `(time_ms, sequence number)`; ties never depend on float equality.
Decisions happen only at a job arrival (switch) and when a policy-routed task enters a
server's ready queue (server). `sim/` never imports `acmad.policies`.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from acmad.interface.actions import NULL_MESSAGE, Message, ServerAction, SwitchAction
from acmad.interface.channel import NULL_CHANNEL, Channel
from acmad.interface.estimators import Estimator, make_estimator
from acmad.interface.observations import FEATURE_NAMES, ExecutorInfo, JobView, ServerObs, SwitchObs, TaskView
from acmad.sim.executor import Executor
from acmad.sim.links import Links, ack_summary
from acmad.sim.load import ArrivalRate
from acmad.sim.models import make_model
from acmad.sim.models.base import TaskFeatures
from acmad.sim.records import OnPeriodRecord, Records, TaskRecord
from acmad.sim.rules import is_allowed, rule_executor
from acmad.sim.server import Server
from acmad.sim.switch import Switch
from acmad.sim.topology import Topology
from acmad.traffic.stream import TaskStream

Tier = Literal["switch", "server"]
_ARRIVAL, _DELIVER, _END, _DEADLINE, _ACK = "arrival", "deliver", "end", "deadline", "ack"


@dataclass(frozen=True)
class SimConfig:
    """Simulator parameters (`configs/sim/*.yaml`)."""

    deadline_ms: float = 1.0          # D: job deadline = t_arrival_ms + D (ADR-0013)
    links: Links = field(default_factory=Links)
    oracle_cluster: bool = False      # fill ServerObs.oracle_cluster (explicit oracle mode, ADR-0014)
    oracle_service: bool = False      # fill ServerObs.oracle_service (explicit oracle mode, ADR-0014)
    estimator: str = "last_report"    # ADR-0014
    load_window_ms: float = 10.0      # window of the arrival rates in c_t and ServerObs (ADR-0014)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "SimConfig":
        cfg = dict(cfg)
        links = Links(**(cfg.pop("links", None) or {}))
        return cls(links=links, **cfg)


@dataclass(frozen=True)
class DecisionRequest:
    """The simulator is paused waiting for `tier`'s action on `obs`."""

    tier: Tier
    server: int | None                # the deciding server (None for the switch)
    obs: SwitchObs | ServerObs
    decision_id: int


@dataclass(slots=True)
class _Task:
    id: int
    job_id: int
    kind: str
    allowed: str
    preds: tuple[int, ...]
    features: TaskFeatures
    status: str = "pending"           # pending|in_flight|blocked|ready|queued|in_service|done
    server: int | None = None
    executor: int | None = None
    t_ready_ms: float | None = None
    t_start_ms: float | None = None
    t_end_ms: float | None = None
    outcome: str | None = None
    energy_dynamic_j: float = 0.0
    server_decision_id: int | None = None


@dataclass(slots=True)
class _Job:
    id: int
    du: int
    arrival_ms: float
    deadline_ms: float
    tasks: list[int]
    server: int | None = None
    message: Message = NULL_MESSAGE
    switch_decision_id: int | None = None
    expired: bool = False


class Simulation:
    """Mechanics of the whole cluster: one switch, `topology.n_servers` servers."""

    def __init__(self, stream: TaskStream, topology: Topology, config: SimConfig = SimConfig(),
                 channel: Channel = NULL_CHANNEL, estimator: Estimator | None = None,
                 seed: int = 0) -> None:
        stream.validate()
        self.config = config
        self.channel = channel
        self.seed = int(seed)          # experiment seed: keys the service noise of every executor (ADR-0002)
        self.horizon_ms = float(stream.meta.get("T_ms", float("inf")))   # episode length T (ADR-0017)
        self.now_ms = 0.0
        self._heap: list[tuple[float, int, str, Any]] = []
        self._seq = itertools.count()
        self._decision_ids = itertools.count()
        self._pending: DecisionRequest | None = None
        self._pending_task: int | None = None
        self._ready_order: deque[int] = deque()        # server ids, one per ready-queue entry
        self.n_decisions = {"switch": 0, "server": 0}

        self.servers = [Server(k, [Executor(s, make_model(s.model_name), seed, k, i) for i, s in enumerate(specs)])
                        for k, specs in enumerate(topology.servers)]
        est = estimator if estimator is not None else make_estimator(config.estimator)
        infos = tuple(tuple(ExecutorInfo(e.kind, e.spec.capacity) for e in s.executors) for s in self.servers)
        self.switch = Switch(infos, [ack_summary(s.executors, 0.0) for s in self.servers], est,
                             downlink_ms=config.links.downlink_ms, load_window_ms=config.load_window_ms)
        self.server_load = [ArrivalRate(config.load_window_ms) for _ in self.servers]   # local arrival rate
        self._load(stream)

    # ------------------------------------------------------------------ setup
    def _load(self, stream: TaskStream) -> None:
        self.tasks: dict[int, _Task] = {}
        self.jobs: dict[int, _Job] = {}
        tasks = stream.tasks.sort_values("task_id", kind="stable")
        by_job: dict[int, list[int]] = {}
        for r in tasks.itertuples(index=False):
            dep = () if pd.isna(r.depends_on) else (int(r.depends_on),)
            t = _Task(int(r.task_id), int(r.job_id), str(r.kind), str(r.allowed), dep,
                      TaskFeatures(str(r.kind), int(r.nprb), int(r.mcs), int(r.snr_db), int(r.tbs), int(r.base_graph),
                                   int(r.n_cb), int(r.cb_len), int(r.total_bits), int(r.qm), float(r.code_rate)))
            self.tasks[t.id] = t
            by_job.setdefault(t.job_id, []).append(t.id)
        jobs = stream.jobs.sort_values(["t_arrival_ms", "job_id"], kind="stable")
        for r in jobs.itertuples(index=False):
            a = float(r.t_arrival_ms)
            j = _Job(int(r.job_id), int(r.du), a, a + self.config.deadline_ms, by_job.get(int(r.job_id), []))
            self.jobs[j.id] = j
            self._push(a, _ARRIVAL, j.id)
            self._push(j.deadline_ms, _DEADLINE, j.id)

    def _push(self, t_ms: float, kind: str, payload: Any) -> None:
        heapq.heappush(self._heap, (float(t_ms), next(self._seq), kind, payload))

    # ------------------------------------------------------------------ loop
    def next_decision(self) -> DecisionRequest | None:
        """Advance until a policy must act; None when the episode is over."""
        while self._pending is None:
            if self._ready_order:
                self._drain_one()
                continue
            if not self._heap:
                return None
            t, _, kind, payload = heapq.heappop(self._heap)
            assert t >= self.now_ms, "event in the past"
            self.now_ms = t
            getattr(self, f"_on_{kind}")(payload)
        return self._pending

    def apply(self, action: SwitchAction | ServerAction) -> None:
        """Execute the action for the pending decision request."""
        req = self._pending
        if req is None:
            raise RuntimeError("no pending decision; call next_decision() first")
        if req.tier == "switch":
            if not isinstance(action, SwitchAction):
                raise TypeError(f"switch needs SwitchAction, got {type(action).__name__}")
            self._dispatch(req.obs.job.id, int(action.server), action.message, req.decision_id)
        else:
            if not isinstance(action, ServerAction):
                raise TypeError(f"server needs ServerAction, got {type(action).__name__}")
            tid = self._pending_task
            task = self.tasks[tid]
            e = int(action.executor)
            execs = self.servers[req.server].executors
            if not 0 <= e < len(execs) or not is_allowed(task.allowed, execs[e].kind):
                raise ValueError(f"executor {e} not allowed for task {tid} ({task.allowed})")
            task.server_decision_id = req.decision_id
            self._assign(tid, e)
            self._pending_task = None
        self._pending = None

    def finalize(self) -> Records:
        """Per-task outcomes; the episode must have run to the end."""
        if self._pending is not None or self._heap or self._ready_order:
            raise RuntimeError("simulation not finished")
        out = []
        for tid in sorted(self.tasks):
            t = self.tasks[tid]
            j = self.jobs[t.job_id]
            if t.outcome is None:
                raise RuntimeError(f"task {tid} has no outcome")   # conservation
            msg = None if j.message.is_null else tuple(float(x) for x in j.message.vector)
            out.append(TaskRecord(
                job_id=j.id, task_id=tid, kind=t.kind, server=j.server, executor=t.executor,
                message=msg, t_arrival_ms=j.arrival_ms, t_ready_ms=t.t_ready_ms,
                t_start_ms=t.t_start_ms, t_end_ms=t.t_end_ms, deadline_ms=j.deadline_ms,
                outcome=t.outcome, energy_dynamic_j=t.energy_dynamic_j,
                on_period=None if t.executor is None else self.servers[j.server].executors[t.executor].period_of.get(tid),
                switch_decision_id=j.switch_decision_id, server_decision_id=t.server_decision_id))
        periods = [OnPeriodRecord(server=srv.id, executor=i, kind=ex.kind, on_period=p.index, t_on_ms=p.t_on_ms,
                                  t_off_ms=p.t_off_ms, static_j=ex.static_power_w * (p.t_off_ms - p.t_on_ms) * 1e-3,
                                  n_tasks=p.n_tasks)
                   for srv in self.servers for i, ex in enumerate(srv.executors)
                   for p in ex.shut_down(max(self.horizon_ms, self.now_ms))]
        return Records(out, periods)

    # ------------------------------------------------------------------ events
    def _on_arrival(self, job_id: int) -> None:
        job = self.jobs[job_id]
        if job.expired:           # only possible with D <= 0
            return
        for tid in job.tasks:
            self.tasks[tid].status = "in_flight"
        self._request("switch", None, self.switch.observe(self.now_ms, self._job_view(job)))

    def _dispatch(self, job_id: int, server: int, message: Message, decision_id: int) -> None:
        job = self.jobs[job_id]
        self.switch.dispatch(server, self.now_ms, sum(self.tasks[t].features.total_bits for t in job.tasks))
        job.server, job.message = server, self.channel.transmit(message)
        job.switch_decision_id = decision_id
        self._push(self.now_ms + self.config.links.downlink_ms, _DELIVER, job_id)

    def _on_deliver(self, job_id: int) -> None:
        job = self.jobs[job_id]
        if job.expired:
            return                 # tasks already dropped at the deadline
        srv = self.servers[job.server]
        if job.tasks:
            self.server_load[srv.id].add(self.now_ms)
        for tid in job.tasks:
            t = self.tasks[tid]
            t.server, t.status = srv.id, "blocked"
        entered = srv.receive(job.tasks, {tid: self.tasks[tid].preds for tid in job.tasks})
        self._entered_ready(srv.id, entered)

    def _entered_ready(self, server: int, task_ids: list[int]) -> None:
        for tid in task_ids:
            t = self.tasks[tid]
            t.status, t.t_ready_ms = "ready", self.now_ms
            self._ready_order.append(server)

    def _drain_one(self) -> None:
        """Bind the head of one server's ready queue: by rule, or ask the server policy."""
        srv = self.servers[self._ready_order.popleft()]
        tid = srv.ready.popleft()
        task, job = self.tasks[tid], self.jobs[self.tasks[tid].job_id]
        if job.expired:           # released after its deadline passed (ADR-0013)
            self._drop(task, 0.0, "dropped")
            return
        e = rule_executor(task.kind, task.allowed, srv.cpu)
        if e is not None:
            self._assign(tid, e)
            return
        views = tuple(ex.view(self.now_ms, is_allowed(task.allowed, ex.kind)) for ex in srv.executors)
        cluster = (np.concatenate([ack_summary(s.executors, self.now_ms).ravel() for s in self.servers])
                   if self.config.oracle_cluster else None)
        service = (tuple(ex.peek(tid, task.features) if v.allowed else None for ex, v in zip(srv.executors, views))
                   if self.config.oracle_service else None)
        obs = ServerObs(now_ms=self.now_ms, server_id=srv.id, task=self._task_view(task, job),
                        executors=views, message=job.message,
                        arrival_rate_per_ms=self.server_load[srv.id].rate_per_ms(self.now_ms),
                        oracle_cluster=cluster, oracle_service=service)
        self._pending_task = tid
        self._request("server", srv.id, obs)

    def _assign(self, tid: int, e: int) -> None:
        task = self.tasks[tid]
        srv = self.servers[task.server]
        task.executor, task.status = e, "queued"
        srv.executors[e].enqueue(tid, task.features, self.now_ms)
        self._start(srv, e)

    def _start(self, srv: Server, e: int) -> None:
        ex = srv.executors[e]
        # A task whose deadline has been reached cannot meet it (sigma > 0): drop it instead of
        # starting it (its deadline event may be later in the heap at this same instant).
        for tid in [t for t in ex.waiting if self.jobs[self.tasks[t].job_id].deadline_ms <= self.now_ms]:
            ex.remove_waiting(tid)
            self._drop(self.tasks[tid], 0.0, "dropped")
        for tid, end in ex.start_next(self.now_ms):
            t = self.tasks[tid]
            t.status, t.t_start_ms = "in_service", self.now_ms
            self._push(end, _END, tid)
        ex.settle(self.now_ms)    # idle from now if nothing is left (ADR-0004)

    def _on_end(self, tid: int) -> None:
        task = self.tasks[tid]
        if task.status != "in_service":
            return                 # stale: aborted at its deadline
        srv = self.servers[task.server]
        task.energy_dynamic_j = srv.executors[task.executor].finish(tid)
        task.status, task.t_end_ms = "done", self.now_ms
        task.outcome = "met" if self.now_ms <= self.jobs[task.job_id].deadline_ms else "missed"
        self._entered_ready(srv.id, srv.complete(tid))
        self._start(srv, task.executor)
        # ACK trigger: every service completion (ADR-0014).
        self._push(self.now_ms + self.config.links.uplink_ms, _ACK,
                   (srv.id, ack_summary(srv.executors, self.now_ms), self.now_ms))

    def _on_ack(self, payload: tuple[int, np.ndarray, float]) -> None:
        self.switch.on_ack(*payload)

    def _on_deadline(self, job_id: int) -> None:
        """ADR-0013: drop what is left of the job; abort tasks in service (proportional energy)."""
        job = self.jobs[job_id]
        job.expired = True
        for tid in job.tasks:
            t = self.tasks[tid]
            if t.status in ("done", "pending"):
                if t.status == "pending":
                    self._drop(t, 0.0, "dropped")
                continue
            if t.status == "in_flight":
                self._drop(t, 0.0, "dropped")
            elif t.status == "blocked":
                self.servers[t.server].blocked.remove(tid)
                self._drop(t, 0.0, "dropped")
            elif t.status == "queued":
                ex = self.servers[t.server].executors[t.executor]
                ex.remove_waiting(tid)
                ex.settle(self.now_ms)
                self._drop(t, 0.0, "dropped")
            elif t.status == "in_service":
                ex = self.servers[t.server].executors[t.executor]
                if ex.in_service[tid].end_ms <= job.deadline_ms:
                    continue       # completes exactly at the deadline: met
                self._drop(t, ex.abort(tid, self.now_ms), "missed")
                self._start(self.servers[t.server], t.executor)
            # "ready" cannot occur: the ready queue is drained before any event.

    def _drop(self, t: _Task, energy_dynamic_j: float, outcome: str) -> None:
        t.status, t.outcome, t.energy_dynamic_j, t.t_end_ms = "done", outcome, energy_dynamic_j, self.now_ms

    # ------------------------------------------------------------------ views
    def _request(self, tier: Tier, server: int | None, obs) -> None:
        self.n_decisions[tier] += 1
        self._pending = DecisionRequest(tier, server, obs, next(self._decision_ids))

    def _task_view(self, t: _Task, job: _Job) -> TaskView:
        f = t.features
        return TaskView(id=t.id, job_id=t.job_id, kind=t.kind,
                        features=np.array([getattr(f, n) for n in FEATURE_NAMES], dtype=np.float64),
                        deadline_ms=job.deadline_ms, slack_ms=job.deadline_ms - self.now_ms)

    def _job_view(self, job: _Job) -> JobView:
        return JobView(id=job.id, du=job.du, arrival_ms=job.arrival_ms, deadline_ms=job.deadline_ms,
                       slack_ms=job.deadline_ms - self.now_ms,
                       tasks=tuple(self._task_view(self.tasks[t], job) for t in job.tasks))
