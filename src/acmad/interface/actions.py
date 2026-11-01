"""Actions returned by policies (ARCHITECTURE §4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Message:
    """The dispatch-header field; paper: preference w_i.

    `vector is None` is the null (preference-free) message. A non-null vector is made
    read-only on construction; its dimension and bit width are enforced by `Channel`.
    """

    vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.vector is not None:
            v = np.array(self.vector, dtype=np.float64)   # own copy
            if v.ndim != 1:
                raise ValueError(f"message vector must be 1-D, got shape {v.shape}")
            v.setflags(write=False)
            object.__setattr__(self, "vector", v)

    @property
    def is_null(self) -> bool:
        return self.vector is None


NULL_MESSAGE = Message(None)


@dataclass(frozen=True)
class SwitchAction:
    """Upstream decision for one job (ADR-0011): target server k and header message."""

    server: int
    message: Message = NULL_MESSAGE


@dataclass(frozen=True)
class ServerAction:
    """Downstream decision for one ready task: executor index e within the server."""

    executor: int
