"""Service model for `gpu` executors: NVIDIA V100, speed p = 1 (`GPU_dataset.csv`); static power = measured floor (ADR-0004)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from acmad.sim.models.base import ExecutorSpec, ServiceModel, TaskFeatures, register_model
from acmad.sim.models.cloudric_ldpc import CloudricLdpcModel


@register_model("gpu")
class GPUModel(CloudricLdpcModel, ServiceModel):
    """`ldpc_decoding` from CloudRIC's measurements (ADR-0003); `cpu_processing` not modelled yet."""

    KIND = "gpu"

    def __call__(self, task: TaskFeatures, spec: ExecutorSpec, rng: np.random.Generator) -> tuple[float, float]:
        if task.kind != "ldpc_decoding":
            raise NotImplementedError(f"{task.kind} on gpu: not modelled yet (ADR-0003)")
        if spec.params.get("speed", 1.0) != 1.0:
            raise NotImplementedError("accelerator speed p != 1: deferred (ADR-0010)")
        return self.ldpc(task, rng)

    def mean_service_ms(self, tasks: pd.DataFrame, spec: ExecutorSpec) -> np.ndarray:
        if spec.params.get("speed", 1.0) != 1.0:
            raise NotImplementedError("accelerator speed p != 1: deferred (ADR-0010)")
        out = np.full(len(tasks), np.nan)
        ld = tasks.kind.to_numpy(str) == "ldpc_decoding"
        if ld.any():
            out[ld] = self.ldpc_mean_ms(tasks[ld])
        return out
