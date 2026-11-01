"""Topology: the servers k = 1..K and their executors E_k, from `configs/topology/*.yaml` (ADR-0010)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from acmad.sim.models.base import T_OFF_MS, ExecutorSpec
from acmad.sim.rules import cpu_index


@dataclass(frozen=True)
class Topology:
    """`servers[k]` is the tuple of executor specs of server k (indices are executor ids)."""

    servers: tuple[tuple[ExecutorSpec, ...], ...]

    def __post_init__(self) -> None:
        if not self.servers:
            raise ValueError("topology needs at least one server")
        for k, execs in enumerate(self.servers):
            try:
                cpu_index(list(execs))
            except ValueError as err:
                raise ValueError(f"server {k}: {err}") from None

    @property
    def n_servers(self) -> int:
        return len(self.servers)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "Topology":
        """`{servers: [{executors: [{kind, capacity, params, model?, t_off_ms?}, ...]}, ...]}`."""
        return cls(tuple(
            tuple(ExecutorSpec(kind=e["kind"], capacity=int(e.get("capacity", 1)),
                               params=dict(e.get("params") or {}), model=e.get("model"),
                               t_off_ms=float(e.get("t_off_ms", T_OFF_MS)))
                  for e in s["executors"])
            for s in cfg["servers"]))


def load_topology(path: str | Path) -> Topology:
    """Read and validate a topology YAML file."""
    return Topology.from_dict(yaml.safe_load(Path(path).read_text()))
