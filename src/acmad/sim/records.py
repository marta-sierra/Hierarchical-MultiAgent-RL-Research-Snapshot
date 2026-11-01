"""Per-task outcome log and executor on-periods: the ground truth for metrics (ARCHITECTURE §7, ADR-0004)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from operator import attrgetter

import pandas as pd

#: `met`: completed with t_end_ms <= deadline_ms. `dropped`: removed before starting service
#: (ADR-0013). `missed`: aborted in service at the deadline, partial energy charged (ADR-0013).
OUTCOMES: tuple[str, ...] = ("met", "missed", "dropped")


@dataclass(frozen=True, slots=True)
class TaskRecord:
    job_id: int
    task_id: int
    kind: str
    server: int | None             # None if the job never reached a server
    executor: int | None           # None if never assigned
    message: tuple[float, ...] | None
    t_arrival_ms: float            # job arrival at the switch
    t_ready_ms: float | None       # entry into the server's ready queue
    t_start_ms: float | None
    t_end_ms: float                # completion, abort or drop time
    deadline_ms: float
    outcome: str
    energy_dynamic_j: float
    on_period: int | None          # on-period of its executor it ran in (None if it never started)
    switch_decision_id: int | None
    server_decision_id: int | None  # None when routed by rule


@dataclass(frozen=True)
class OnPeriodRecord:
    """One on-period of an executor (ADR-0004): static energy = P_static * (t_off - t_on)."""

    server: int
    executor: int
    kind: str
    on_period: int                 # per executor, 0, 1, ... (= number of switch-ons - 1)
    t_on_ms: float
    t_off_ms: float                # last idle start + T_off, clipped at the episode end max(T, last event)
    static_j: float
    n_tasks: int                   # tasks that started service in it


class Records(tuple):
    """Tuple of `TaskRecord`, ordered by task_id; `on_periods` holds the executors' on-periods."""

    def __new__(cls, tasks=(), on_periods=()):
        self = super().__new__(cls, tasks)
        self.on_periods = tuple(on_periods)
        return self

    def to_frame(self) -> pd.DataFrame:
        cols = list(TaskRecord.__dataclass_fields__)
        get = attrgetter(*cols)
        return pd.DataFrame([get(r) for r in self], columns=cols)     # no deep copy (asdict)

    def on_periods_frame(self) -> pd.DataFrame:
        cols = list(OnPeriodRecord.__dataclass_fields__)
        return pd.DataFrame([asdict(p) for p in self.on_periods], columns=cols)
