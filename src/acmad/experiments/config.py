"""Experiment configuration: methods x scenarios x seeds (configs/experiment/*.yaml).

    name: benchmark
    seeds: [0, 1, 2]                       # experiment seeds x (ADR-0002)
    split: test                            # traffic split (ADR-0017)
    topology: configs/topology/example.yaml
    sim: configs/sim/default.yaml          # base simulator config ...
    sim_overrides: {deadline_ms: 3.0}      # ... and optional overrides
    traffic: configs/traffic/default.yaml
    scenarios:                             # grid: every combination of the lists ...
      n_dus: [1, 3]
      s: [1.0, 2.0]
      load_class: [null]                   # null = any class of the split (ADR-0007)
    # ... or a path to such a grid; omitted: configs/scenarios/grid.yaml (the 28-scenario grid)
    methods:
      - {name: rr, switch: round_robin, server: round_robin}
      - {name: ws, switch: least_loaded, server: {name: weighted_score, alpha: 0.5}}
    save_records: true                     # records.parquet per cell (large)
    check_invariants: false                # run acmad.sim.checks on every cell

A policy is a name or `{name: ..., <constructor kwargs>}`; `server` may also be a list with one entry
per server. Relative paths resolve against the repository root.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from acmad import cache
from acmad.sim import SimConfig, Topology

REPO = Path(__file__).resolve().parents[3]


def _path(p: str | Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO / q


DEFAULT_SCENARIOS = "configs/scenarios/grid.yaml"   # the 28-scenario benchmark grid (ADR-0021)

@dataclass(frozen=True)
class PolicySpec:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, x: str | dict) -> "PolicySpec":
        if isinstance(x, str):
            return cls(x)
        x = dict(x)
        return cls(x.pop("name"), x)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, **self.kwargs}


@dataclass(frozen=True)
class Method:
    """One decision-maker: the switch policy and a server policy for every server."""

    name: str
    switch: PolicySpec
    servers: tuple[PolicySpec, ...] | PolicySpec   # one for all servers, or one per server

    def server_specs(self, n_servers: int) -> tuple[PolicySpec, ...]:
        if isinstance(self.servers, PolicySpec):
            return (self.servers,) * n_servers
        if len(self.servers) != n_servers:
            raise ValueError(f"method {self.name}: {len(self.servers)} server policies for {n_servers} servers")
        return self.servers

    @classmethod
    def parse(cls, d: dict) -> "Method":
        srv = d["server"]
        servers = tuple(PolicySpec.parse(x) for x in srv) if isinstance(srv, list) else PolicySpec.parse(srv)
        return cls(str(d["name"]), PolicySpec.parse(d["switch"]), servers)

    def to_dict(self) -> dict[str, Any]:
        srv = self.servers.to_dict() if isinstance(self.servers, PolicySpec) else [s.to_dict() for s in self.servers]
        return {"name": self.name, "switch": self.switch.to_dict(), "server": srv}


@dataclass(frozen=True)
class Scenario:
    """Traffic condition of an episode (ADR-0017 knobs, ADR-0007 load class)."""

    n_dus: int
    s: float
    load_class: str | None = None

    @property
    def id(self) -> str:
        return f"n{self.n_dus}_s{self.s:g}_{self.load_class or 'any'}"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seeds: tuple[int, ...]
    split: str
    topology_path: Path
    topology: Topology
    topology_dict: dict[str, Any]
    sim: SimConfig
    sim_dict: dict[str, Any]
    traffic_path: Path
    scenarios: tuple[Scenario, ...]
    methods: tuple[Method, ...]
    save_records: bool = True
    check_invariants: bool = False

    def resolved(self) -> dict[str, Any]:
        """Everything a run depends on, as saved in `config.yaml` (ARCHITECTURE §7)."""
        from acmad.traffic import load_config
        return {"name": self.name, "seeds": list(self.seeds), "split": self.split,
                "topology_path": str(self.topology_path.relative_to(REPO) if self.topology_path.is_relative_to(REPO)
                                     else self.topology_path),
                "topology": self.topology_dict, "sim": self.sim_dict,
                "traffic": load_config(self.traffic_path).to_dict(),
                "scenarios": [vars(s) for s in self.scenarios], "methods": [m.to_dict() for m in self.methods],
                "save_records": self.save_records, "check_invariants": self.check_invariants}

    def hash(self) -> str:
        return cache.key(self.resolved())


def load_experiment(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(_path(path).read_text())
    topo_path = _path(raw.get("topology", "configs/topology/example.yaml"))
    topo_dict = yaml.safe_load(topo_path.read_text())
    sim_dict = yaml.safe_load(_path(raw.get("sim", "configs/sim/default.yaml")).read_text()) or {}
    for k, v in (raw.get("sim_overrides") or {}).items():
        sim_dict[k] = {**sim_dict.get(k, {}), **v} if isinstance(v, dict) else v
    sc = raw.get("scenarios", DEFAULT_SCENARIOS)
    if isinstance(sc, str):
        sc = yaml.safe_load(_path(sc).read_text())
    scenarios = tuple(Scenario(int(n), float(s), lc) for n, s, lc in
                      itertools.product(sc["n_dus"], sc["s"], sc.get("load_class", [None])))
    methods = tuple(Method.parse(m) for m in raw["methods"])
    names = [m.name for m in methods]
    if len(set(names)) != len(names):
        raise ValueError(f"method names must be unique: {names}")
    return ExperimentConfig(
        name=str(raw["name"]), seeds=tuple(int(x) for x in raw["seeds"]), split=str(raw.get("split", "test")),
        topology_path=topo_path, topology=Topology.from_dict(topo_dict), topology_dict=topo_dict,
        sim=SimConfig.from_dict(sim_dict), sim_dict=sim_dict,
        traffic_path=_path(raw.get("traffic", "configs/traffic/default.yaml")),
        scenarios=scenarios, methods=methods, save_records=bool(raw.get("save_records", True)),
        check_invariants=bool(raw.get("check_invariants", False)))
