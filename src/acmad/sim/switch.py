"""Switch u (mechanics): job arrivals, offered load, dispatch record, last ACK per server (ADR-0011, ADR-0014)."""
from __future__ import annotations

from collections import deque

import numpy as np

from acmad.interface.estimators import Estimator
from acmad.interface.observations import FEATURE_NAMES, ClusterState, ExecutorInfo, JobView, ServerReport, SwitchObs
from acmad.sim.load import ArrivalRate, MeanTBS

_TBS = FEATURE_NAMES.index("tbs")


class Switch:
    """What the upstream dispatcher knows: the topology, what it measures itself (c_t) and the
    ACKs it received.

    Nothing here reads server internals; server state reaches the switch only through
    `on_ack` (ServerReport, aged by the uplink).
    """

    def __init__(self, topology: tuple[tuple[ExecutorInfo, ...], ...], initial_summaries: list[np.ndarray],
                 estimator: Estimator | None, downlink_ms: float = 0.0, load_window_ms: float = 10.0) -> None:
        self.topology = topology
        self.n_servers = len(topology)
        self.estimator = estimator
        self.downlink_ms = float(downlink_ms)
        self.arrivals = ArrivalRate(load_window_ms)
        self.tbs = MeanTBS()
        # Before any ACK the switch holds the (empty) state at sim start, t = 0.
        self._last: list[tuple[np.ndarray, float]] = [(s, 0.0) for s in initial_summaries]
        # Per server: (delivery time, bits) of each job dispatched there that its newest report cannot reflect.
        self._unacked: list[deque[tuple[float, int]]] = [deque() for _ in range(self.n_servers)]

    def cluster_state(self, now_ms: float) -> ClusterState:
        """c_t (ADR-0014): offered load + per-server dispatch record since the newest report."""
        return ClusterState(arrival_rate_per_ms=self.arrivals.rate_per_ms(now_ms), mean_tbs=self.tbs.value,
                            jobs_since_ack=np.array([len(q) for q in self._unacked], dtype=np.float64),
                            bits_since_ack=np.array([sum(b for _, b in q) for q in self._unacked], dtype=np.float64))

    def on_ack(self, server: int, summary: np.ndarray, reported_at_ms: float) -> None:
        """Keep the most recent report per server (ACKs may arrive out of order)."""
        if reported_at_ms >= self._last[server][1]:
            self._last[server] = (summary, reported_at_ms)
            q = self._unacked[server]
            while q and q[0][0] <= reported_at_ms:     # delivered by then: the report reflects it
                q.popleft()

    def reports(self, now_ms: float) -> tuple[ServerReport, ...]:
        return tuple(ServerReport(s, t, now_ms - t) for s, t in self._last)

    def observe(self, now_ms: float, job: JobView) -> SwitchObs:
        """A job arrives: update the offered load, then build the switch observation."""
        self.arrivals.add(now_ms)
        self.tbs.update(job.tasks[0].features[_TBS] if job.tasks else 0.0)
        reps = self.reports(now_ms)
        est = self.estimator.estimate(reps, now_ms) if self.estimator is not None else None
        return SwitchObs(now_ms=now_ms, job=job, cluster=self.cluster_state(now_ms), reports=reps,
                         topology=self.topology, estimates=est)

    def dispatch(self, server: int, now_ms: float, bits: int) -> None:
        if not 0 <= server < self.n_servers:
            raise ValueError(f"switch chose server {server}, have {self.n_servers}")
        self._unacked[server].append((now_ms + self.downlink_ms, int(bits)))
