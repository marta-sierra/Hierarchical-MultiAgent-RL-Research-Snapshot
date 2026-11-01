"""Executor e: one FIFO queue in front of `capacity` parallel units, non-preemptive (ADR-0010);
switched on by work and off after `T_off` idle (ADR-0004)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from acmad.interface.observations import ExecutorView
from acmad.sim.models.base import ExecutorSpec, ServiceModel, TaskFeatures

SERVICE_STREAM = 0        # namespace of the service-noise keys in the experiment's SeedSequence (ADR-0002)


@dataclass(slots=True)
class _InService:
    start_ms: float
    end_ms: float


@dataclass
class OnPeriod:
    """One on-period of an executor (ADR-0004): from switch-on to switch-off."""

    index: int                     # per executor, 0, 1, ...
    t_on_ms: float
    t_off_ms: float | None = None  # None while on
    n_tasks: int = 0               # tasks that started service in it


class Executor:
    """Mechanics of one executor. Service times and energies come from its model (ADR-0015).

    The model is called once per task, when the task is enqueued here; its rng is keyed by
    `(seed; SERVICE_STREAM, server, executor, task)`, so a task's draw on an executor does not
    depend on call order or on any decision. Policies never see these values (ADR-0001).

    On / off (ADR-0004): the executor starts off. Work arriving while it is off switches it on
    at once (0 latency, a new on-period). When it has no task in service or queued it is idle;
    it switches off `spec.t_off_ms` after becoming idle unless work arrives first (work arriving
    exactly at that instant finds it off). Off times are settled lazily, with no events.
    """

    def __init__(self, spec: ExecutorSpec, model: ServiceModel, seed: int = 0, server: int = 0,
                 index: int = 0) -> None:
        self.spec = spec
        self.model = model
        self._key = (int(seed), (SERVICE_STREAM, int(server), int(index)))
        self.waiting: deque[int] = deque()
        self.in_service: dict[int, _InService] = {}
        self._cost: dict[int, tuple[float, float]] = {}    # task_id -> (sigma, energy)
        self._bits: dict[int, int] = {}                     # task_id -> total_bits (observable)
        self.periods: list[OnPeriod] = []
        self._idle_since: float | None = None              # None while busy or off
        self.period_of: dict[int, int] = {}                 # task_id -> on-period it started in

    @property
    def kind(self) -> str:
        return self.spec.kind

    @property
    def n_free(self) -> int:
        return self.spec.capacity - len(self.in_service)

    def rng(self, task_id: int) -> np.random.Generator:
        """The service-noise generator of task `task_id` on this executor."""
        seed, (stream, server, index) = self._key
        return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(stream, server, index, int(task_id))))

    @property
    def static_power_w(self) -> float:
        return float(self.model.static_power_w)

    def is_on(self, now_ms: float) -> bool:
        if not self.periods or self.periods[-1].t_off_ms is not None:
            return False
        return self._idle_since is None or now_ms - self._idle_since < self.spec.t_off_ms

    def _wake(self, now_ms: float) -> None:
        """Work arrives: switch on if off, otherwise leave the idle state."""
        if not self.is_on(now_ms):
            self._close(now_ms)
            self.periods.append(OnPeriod(len(self.periods), now_ms))
        self._idle_since = None

    def _close(self, now_ms: float) -> None:
        """Record the switch-off of an on-period that has run out its idle time."""
        if self.periods and self.periods[-1].t_off_ms is None and self._idle_since is not None:
            self.periods[-1].t_off_ms = self._idle_since + self.spec.t_off_ms
            assert self.periods[-1].t_off_ms <= now_ms

    def shut_down(self, end_ms: float = float("inf")) -> list[OnPeriod]:
        """End of the episode at `end_ms`: the executor must be idle; its last period closes T_off after
        going idle, but never after `end_ms` (nothing is charged past the episode, ADR-0004)."""
        if self.in_service or self.waiting:
            raise RuntimeError("executor still has work")
        self._close(float("inf"))
        if self.periods and self.periods[-1].t_off_ms > end_ms:
            self.periods[-1].t_off_ms = max(end_ms, self.periods[-1].t_on_ms)
        return self.periods

    def peek(self, task_id: int, features: TaskFeatures) -> tuple[float, float]:
        """`(service_time_ms, dynamic_energy_j)` the task gets if enqueued here: the same keyed draw
        as `enqueue`, without side effects. Only for the explicit service oracle (ADR-0014)."""
        s, e = self.model(features, self.spec, self.rng(task_id))
        return float(s), float(e)

    def enqueue(self, task_id: int, features: TaskFeatures, now_ms: float) -> None:
        """Append to the FIFO tail and draw sigma(e, i), e(a) for the task; switches on if off."""
        self._wake(now_ms)
        s, e = self.peek(task_id, features)
        if not s > 0 or e < 0:
            raise ValueError(f"model returned service_time_ms={s}, dynamic_energy_j={e}")
        self._cost[task_id] = (s, e)
        self._bits[task_id] = int(features.total_bits)
        self.waiting.append(task_id)

    def start_next(self, now_ms: float) -> list[tuple[int, float]]:
        """Start FIFO head tasks on free units; returns `(task_id, end_ms)` for each."""
        started = []
        while self.waiting and self.n_free > 0:
            tid = self.waiting.popleft()
            end = now_ms + self._cost[tid][0]
            self.in_service[tid] = _InService(now_ms, end)
            self.periods[-1].n_tasks += 1
            self.period_of[tid] = self.periods[-1].index
            started.append((tid, end))
        return started

    def _release(self, task_id: int) -> tuple[float, float]:
        self._bits.pop(task_id, None)
        return self._cost.pop(task_id)

    def finish(self, task_id: int) -> float:
        """Task completed service; frees its unit. Returns its full dynamic energy (ADR-0004)."""
        del self.in_service[task_id]
        return self._release(task_id)[1]

    def settle(self, now_ms: float) -> None:
        """After the engine has started what it could: idle from `now_ms` if nothing is in
        service or queued (the T_off countdown starts)."""
        if not self.in_service and not self.waiting and self._idle_since is None:
            self._idle_since = now_ms

    def remove_waiting(self, task_id: int) -> None:
        """Drop a task still in the FIFO (no energy, ADR-0013)."""
        self.waiting.remove(task_id)
        self._release(task_id)

    def abort(self, task_id: int, now_ms: float) -> float:
        """Abort a task in service (ADR-0013); frees its unit at once.

        Returns the dynamic energy charged: dynamic_energy_j * elapsed_ms / service_time_ms.
        """
        run = self.in_service.pop(task_id)
        s, e = self._release(task_id)
        return e * min(max(now_ms - run.start_ms, 0.0), s) / s

    def state(self, now_ms: float) -> tuple[int, int, float, int]:
        """Observable state (ADR-0001): `(n_busy, n_queued, elapsed_ms, queued_bits)`."""
        elapsed = sum(max(now_ms - r.start_ms, 0.0) for r in self.in_service.values())
        return len(self.in_service), len(self.waiting), float(elapsed), sum(self._bits[t] for t in self.waiting)

    def view(self, now_ms: float, allowed: bool) -> ExecutorView:
        """Local state of this executor, as a policy may see it (no service times, ADR-0001)."""
        n_busy, n_queued, elapsed, bits = self.state(now_ms)
        return ExecutorView(kind=self.kind, capacity=self.spec.capacity, n_busy=n_busy, n_queued=n_queued,
                            elapsed_ms=elapsed, queued_bits=bits, allowed=allowed)
