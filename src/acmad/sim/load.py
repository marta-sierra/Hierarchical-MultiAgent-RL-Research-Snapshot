"""Offered load measured where jobs arrive: at the switch (c_t) and at each server (ADR-0014)."""
from __future__ import annotations

from collections import deque


class ArrivalRate:
    """Job arrival rate over a sliding time window: jobs arrived in (now - W, now] divided by W (jobs/ms).

    Before W has elapsed the count is still divided by W (the rate ramps up from 0 over the first W).
    """

    def __init__(self, window_ms: float) -> None:
        if not window_ms > 0:
            raise ValueError("window_ms must be > 0")
        self.window_ms = float(window_ms)
        self._times: deque[float] = deque()

    def add(self, now_ms: float) -> None:
        self._times.append(now_ms)

    def rate_per_ms(self, now_ms: float) -> float:
        start = now_ms - self.window_ms
        while self._times and self._times[0] <= start:
            self._times.popleft()
        return len(self._times) / self.window_ms


class MeanTBS:
    """EWMA of the jobs' TBS, b <- 0.9 b + 0.1 TBS; 0 before the first job, which sets it to its own TBS."""

    ALPHA = 0.1

    def __init__(self) -> None:
        self.value = 0.0
        self._started = False

    def update(self, tbs: float) -> None:
        if self._started:
            self.value += self.ALPHA * (tbs - self.value)
        else:
            self.value, self._started = float(tbs), True
