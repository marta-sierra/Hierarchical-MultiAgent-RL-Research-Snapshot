"""Per-executor service-time and energy models, sigma(e, i) and e(a) (ADR-0015)."""
from acmad.sim.models import cpu, gpu, synthetic  # noqa: F401  (register the kinds and the validation model)
from acmad.sim.models.base import (
    EXECUTOR_KINDS, ExecutorSpec, ServiceModel, TaskFeatures, available_models, make_model,
    register_model,
)

__all__ = ["EXECUTOR_KINDS", "ExecutorSpec", "ServiceModel", "TaskFeatures",
           "available_models", "make_model", "register_model"]
