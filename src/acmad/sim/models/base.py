"""Model interface for sigma(e, i) (service time) and e(a) (energy) (ADR-0015)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

EXECUTOR_KINDS: tuple[str, ...] = ("cpu", "gpu")
T_OFF_MS = 10_000.0             # ADR-0004: idle time before an executor is released (switched off): 10 s


@dataclass(frozen=True, slots=True)
class TaskFeatures:
    """What a model may use about task i (TaskStream feature columns, ADR-0016)."""

    kind: str
    nprb: int
    mcs: int
    snr_db: int
    tbs: int
    base_graph: int
    n_cb: int
    cb_len: int
    total_bits: int
    qm: int
    code_rate: float


@dataclass(frozen=True)
class ExecutorSpec:
    """Executor e as configured in the topology (ADR-0010). `model` names the registered model.

    `t_off_ms`: the executor is released (switched off) after this long without work (ADR-0004).
    """

    kind: str
    capacity: int = 1
    params: dict[str, Any] = field(default_factory=dict)
    model: str | None = None      # None -> the model registered under `kind`
    t_off_ms: float = T_OFF_MS

    def __post_init__(self) -> None:
        if self.kind not in EXECUTOR_KINDS:
            raise ValueError(f"unknown executor kind {self.kind!r}; kinds: {EXECUTOR_KINDS}")
        if int(self.capacity) < 1:
            raise ValueError("capacity must be >= 1")
        if not float(self.t_off_ms) > 0:
            raise ValueError("t_off_ms must be > 0")

    @property
    def model_name(self) -> str:
        return self.model or self.kind


class ServiceModel(ABC):
    """Returns `(service_time_ms, dynamic_energy_j)` for task i on executor e (sigma(e, i) and the
    task's dynamic energy, ADR-0004). `static_power_w` is the executor kind's static power,
    accounted per on-period by the executor, not per task.

    `rng` is keyed by (experiment seed, server, executor, task) (ADR-0002): the model's only randomness.
    """

    static_power_w: float = 0.0

    @abstractmethod
    def __call__(self, task: TaskFeatures, spec: ExecutorSpec, rng: np.random.Generator) -> tuple[float, float]: ...

    def mean_service_ms(self, tasks: "pd.DataFrame", spec: ExecutorSpec) -> np.ndarray:
        """E[sigma(e, i)] in closed form for every row of a TaskStream `tasks` frame; NaN where this
        executor cannot run the task. For experiment design only (offered load rho, ADR-0009);
        never shown to policies (ADR-0001)."""
        raise NotImplementedError(f"{type(self).__name__}: no closed-form mean service time")


_MODELS: dict[str, type[ServiceModel]] = {}
M = TypeVar("M", bound=type)


def register_model(name: str) -> Callable[[M], M]:
    """Class decorator: make a model selectable as `executor.model: <name>` in the topology."""

    def deco(cls: M) -> M:
        if not issubclass(cls, ServiceModel):
            raise TypeError(f"{cls.__name__} is not a ServiceModel")
        prev = _MODELS.get(name)
        if prev is not None and prev is not cls:
            raise ValueError(f"model {name!r} already registered by {prev.__name__}")
        _MODELS[name] = cls
        return cls

    return deco


def make_model(name: str) -> ServiceModel:
    """Instantiate a registered model."""
    try:
        return _MODELS[name]()
    except KeyError:
        raise KeyError(f"no service model {name!r}; available: {sorted(_MODELS)}") from None


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODELS))
