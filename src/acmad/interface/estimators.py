"""Switch-side signal aging: turn aged ACK reports into state estimates (ADR-0014, provisional)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from acmad.interface.observations import ServerReport


class Estimator(ABC):
    """Maps the last report of every server, at time now_ms, to `SwitchObs.estimates`."""

    @abstractmethod
    def estimate(self, reports: tuple[ServerReport, ...], now_ms: float) -> tuple[np.ndarray, ...] | None: ...


class NoEstimate(Estimator):
    """`none`: no estimate; the switch policy sees only the raw reports."""

    def estimate(self, reports, now_ms):
        return None


class LastReport(Estimator):
    """`last_report`: the estimate is the last reported summary, unchanged (no projection)."""

    def estimate(self, reports, now_ms):
        return tuple(r.summary for r in reports)


ESTIMATORS: dict[str, type[Estimator]] = {
    "none": NoEstimate, "last_report": LastReport,
}


def make_estimator(name: str) -> Estimator:
    """Estimator by config name: `none | last_report`."""
    try:
        return ESTIMATORS[name]()
    except KeyError:
        raise KeyError(f"unknown estimator {name!r}; available: {sorted(ESTIMATORS)}") from None
