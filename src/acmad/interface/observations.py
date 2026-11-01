"""Frozen observation objects: the only thing policies ever see (ARCHITECTURE §2, §4).

All arrays are copied and made read-only on construction. Times in ms (ADR-0012).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from acmad.interface.actions import Message

#: Order of the entries of `TaskView.features` (TaskStream feature columns, ADR-0016).
FEATURE_NAMES: tuple[str, ...] = ("nprb", "mcs", "snr_db", "tbs", "base_graph", "n_cb", "cb_len", "total_bits",
                                  "qm", "code_rate")


def _frozen(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    a = np.array(x, dtype=np.float64)
    a.setflags(write=False)
    return a


class _ReadOnlyArrays:
    """Mixin: copy every ndarray field into a read-only float64 array."""

    def __post_init__(self) -> None:
        for f in fields(self):  # type: ignore[arg-type]
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                object.__setattr__(self, f.name, _frozen(v))
            elif isinstance(v, tuple) and v and all(isinstance(e, np.ndarray) for e in v):
                object.__setattr__(self, f.name, tuple(_frozen(e) for e in v))


@dataclass(frozen=True)
class TaskView(_ReadOnlyArrays):
    """One task i. `features` follows `FEATURE_NAMES`; `slack_ms` = deadline_ms - now_ms."""

    id: int
    job_id: int
    kind: str
    features: np.ndarray
    deadline_ms: float
    slack_ms: float


@dataclass(frozen=True)
class JobView:
    """One job (grant chain) as the switch sees it at arrival (ADR-0011)."""

    id: int
    du: int
    arrival_ms: float
    deadline_ms: float
    slack_ms: float
    tasks: tuple[TaskView, ...]


@dataclass(frozen=True)
class ExecutorView:
    """Local state of one executor e in E_k (part of l_t^k), only what a real system observes (ADR-0001).

    No service time or energy is shown: the policy decides from the task's features.
    - `n_busy`: units currently serving (0..capacity).
    - `n_queued`: tasks waiting in the FIFO (not yet in service).
    - `elapsed_ms`: sum over the tasks in service of the time they have been served so far.
    - `queued_bits`: sum of `total_bits` (LDPC input, ADR-0019) of the waiting tasks.
    - `allowed`: whether the offered task may run on this executor.
    """

    kind: str
    capacity: int
    n_busy: int
    n_queued: int
    elapsed_ms: float
    queued_bits: int
    allowed: bool = True


@dataclass(frozen=True)
class ExecutorInfo:
    """Static description of one executor e (the switch knows the topology, ADR-0014)."""

    kind: str
    capacity: int


#: Columns of `ServerReport.summary`, one row per executor (ADR-0014).
SUMMARY_FIELDS: tuple[str, ...] = ("n_busy", "n_queued", "elapsed_ms", "queued_bits")


@dataclass(frozen=True)
class ServerReport(_ReadOnlyArrays):
    """ACK summary hat{l}^k with its age Delta t^k = now_ms - reported_at_ms (ADR-0014).

    `summary` has one row per executor of the server: `(n_busy, n_queued, elapsed_ms, queued_bits)`.
    """

    summary: np.ndarray
    reported_at_ms: float
    age_ms: float


@dataclass(frozen=True)
class ClusterState(_ReadOnlyArrays):
    """Cluster-level state c_t: what the switch measures itself, never aged (ADR-0014).

    - `arrival_rate_per_ms`: jobs reaching the switch per ms over the last `load_window_ms` (sliding
      window); `mean_tbs`: EWMA of their TBS, b <- 0.9 b + 0.1 TBS.
    - `jobs_since_ack[k]`, `bits_since_ack[k]`: jobs dispatched to server k that its newest report
      cannot reflect yet (delivered after `reported_at_ms`), and the sum of their tasks' `total_bits`.
    """

    arrival_rate_per_ms: float
    mean_tbs: float
    jobs_since_ack: np.ndarray
    bits_since_ack: np.ndarray

    def vector(self) -> np.ndarray:
        """Flat c_t: `[arrival_rate_per_ms, mean_tbs, jobs_since_ack..., bits_since_ack...]`."""
        return np.concatenate([[self.arrival_rate_per_ms, self.mean_tbs], self.jobs_since_ack, self.bits_since_ack])


@dataclass(frozen=True)
class SwitchObs(_ReadOnlyArrays):
    """Upstream observation: cluster-level state c_t, aged reports, estimator output, topology."""

    now_ms: float
    job: JobView
    cluster: ClusterState
    reports: tuple[ServerReport, ...]
    topology: tuple[tuple[ExecutorInfo, ...], ...]
    estimates: tuple[np.ndarray, ...] | None = None


@dataclass(frozen=True)
class ServerObs(_ReadOnlyArrays):
    """Downstream observation for one ready task: local state l_t^k plus the header message.

    `arrival_rate_per_ms`: jobs reaching this server per ms over the last `load_window_ms` (ADR-0014).
    The oracle fields are set only in their explicit oracle modes (ADR-0014):
    - `oracle_cluster`: current, un-aged summary of every server, concatenated;
    - `oracle_service[e]`: `(service_time_ms, dynamic_energy_j)` the task will get on executor e,
      exactly (the draw its model will make there), None where it is not allowed.
    """

    now_ms: float
    server_id: int
    task: TaskView
    executors: tuple[ExecutorView, ...]
    message: Message
    arrival_rate_per_ms: float = 0.0
    oracle_cluster: np.ndarray | None = None
    oracle_service: tuple[tuple[float, float] | None, ...] | None = None
