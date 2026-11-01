"""Compatibility routing that bypasses policies (ARCHITECTURE §1, ADR-0011)."""
from __future__ import annotations

from acmad.sim.models.base import ExecutorSpec

RULE_KINDS: frozenset[str] = frozenset({"cpu_processing"})


def cpu_index(executors: list[ExecutorSpec]) -> int:
    """Index of the server's single CPU executor; raise if there is not exactly one (ADR-0011)."""
    idx = [i for i, e in enumerate(executors) if e.kind == "cpu"]
    if len(idx) != 1:
        # TODO(ADR-0011): rule for servers with several CPU executors or none is open.
        raise ValueError(f"each server needs exactly one cpu executor, got {len(idx)}")
    return idx[0]


def rule_executor(kind: str, allowed: str, cpu: int) -> int | None:
    """Executor fixed by rule, or None if the task is a server-policy decision.

    `cpu_processing` (and any task with `allowed == "cpu"`) goes to the CPU executor.
    """
    if kind in RULE_KINDS or allowed == "cpu":
        return cpu
    return None


def is_allowed(allowed: str, executor_kind: str) -> bool:
    """Whether the TaskStream `allowed` value permits an executor kind."""
    return allowed == "any" or allowed == executor_kind
