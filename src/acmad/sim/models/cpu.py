"""Service model for `cpu` executors: `ldpc_decoding` = FlexRAN on one Xeon Gold 6240R core
(`CPU_dataset.csv`, ADR-0003); `cpu_processing` = MATLAB measurement calibrated to it (ADR-0008);
static power 0 (ADR-0004)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from acmad.sim.models import matlab_cpu_processing
from acmad.sim.models.base import ExecutorSpec, ServiceModel, TaskFeatures, register_model
from acmad.sim.models.cloudric_ldpc import CloudricLdpcModel


@register_model("cpu")
class CPUModel(CloudricLdpcModel, ServiceModel):
    """`ldpc_decoding` from CloudRIC's measurements (ADR-0003); `cpu_processing` from the MATLAB
    measurement (ADR-0008), energy 0."""

    KIND = "cpu"

    def __call__(self, task: TaskFeatures, spec: ExecutorSpec, rng: np.random.Generator) -> tuple[float, float]:
        if task.kind == "cpu_processing":
            return matlab_cpu_processing.load().draw(task, rng), 0.0
        if task.kind != "ldpc_decoding":
            raise NotImplementedError(f"{task.kind} on cpu: no model")
        return self.ldpc(task, rng)

    def mean_service_ms(self, tasks: pd.DataFrame, spec: ExecutorSpec) -> np.ndarray:
        out = np.full(len(tasks), np.nan)
        kind = tasks.kind.to_numpy(str)
        cp, ld = kind == "cpu_processing", kind == "ldpc_decoding"
        if cp.any():
            t = tasks[cp]
            out[cp] = matlab_cpu_processing.load().mean_ms(t.nprb.to_numpy(float), t.qm.to_numpy(float),
                                                           t.total_bits.to_numpy(float))
        if ld.any():
            out[ld] = self.ldpc_mean_ms(tasks[ld])
        return out
