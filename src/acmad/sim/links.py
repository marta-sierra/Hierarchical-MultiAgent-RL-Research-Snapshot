"""Switch <-> server links and the ACK summary builder (ADR-0014)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acmad.interface.observations import SUMMARY_FIELDS
from acmad.sim.executor import Executor


@dataclass(frozen=True)
class Links:
    """Dispatch (downlink) and ACK (uplink) latencies, ms."""

    downlink_ms: float = 0.0   # experiments use 0.1 (configs/sim/default.yaml, ADR-0014)
    uplink_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.downlink_ms < 0 or self.uplink_ms < 0:
            raise ValueError("link latencies must be >= 0")




def ack_summary(executors: list[Executor], now_ms: float) -> np.ndarray:
    """ACK summary hat{l}^k: per executor the observable state `SUMMARY_FIELDS` (ADR-0001), shape (|E_k|, 4).

    An ACK is emitted each time a task completes service on the server, not on drops/aborts (ADR-0014).
    """
    return np.array([e.state(now_ms) for e in executors], dtype=np.float64).reshape(len(executors),
                                                                                   len(SUMMARY_FIELDS))
