"""`synthetic`: textbook service times for validating the mechanics against queueing theory
(docs/SIM_VALIDATION.md). Not a model of any real executor.

`params["dist"]`: "exp" (exponential) or "det" (deterministic); `params["mean"][task.kind]`: mean
service time in ms. Drawn with the task's keyed rng (ADR-0002). No energy, no static power.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from acmad.sim.models.base import ExecutorSpec, ServiceModel, TaskFeatures, register_model

DISTS = ("exp", "det")


@register_model("synthetic")
class SyntheticModel(ServiceModel):
    def __call__(self, task: TaskFeatures, spec: ExecutorSpec, rng: np.random.Generator) -> tuple[float, float]:
        dist, mean = spec.params["dist"], float(spec.params["mean"][task.kind])
        if dist not in DISTS:
            raise ValueError(f"synthetic: dist must be one of {DISTS}, got {dist!r}")
        return (float(rng.exponential(mean)) if dist == "exp" else mean), 0.0

    def mean_service_ms(self, tasks: pd.DataFrame, spec: ExecutorSpec) -> np.ndarray:
        return tasks.kind.astype(str).map(spec.params["mean"]).to_numpy(float)
