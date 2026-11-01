"""Test-only helpers: hand-built TaskStreams, a deterministic service model, trivial policies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from acmad.interface import (
    NULL_MESSAGE, Message, ServerAction, ServerPolicy, SwitchAction, SwitchPolicy,
)
from acmad.sim import SimConfig, Simulation, Topology
from acmad.sim.models import ExecutorSpec, ServiceModel, TaskFeatures, register_model
from acmad.experiments.validation import synthetic_stream
from acmad.traffic.stream import TaskStream

DEFAULT_SVC = {"cpu_processing": 0.5, "ldpc_decoding": 1.0}


@register_model("test_fixed")
class FixedModel(ServiceModel):
    """Test only: service time from `params["svc"][task.kind]` (+ `params["per_prb"]` * nprb),
    energy = `params["power_w"]` * time (mJ-free toy numbers)."""

    def __call__(self, task: TaskFeatures, spec: ExecutorSpec, rng: np.random.Generator) -> tuple[float, float]:
        p = spec.params
        s = p.get("svc", DEFAULT_SVC)[task.kind] + p.get("per_prb", 0.0) * task.nprb
        return s, p.get("power_w", 1.0) * s

    def mean_service_ms(self, tasks: pd.DataFrame, spec: ExecutorSpec) -> np.ndarray:
        p = spec.params
        return tasks.kind.astype(str).map(p.get("svc", DEFAULT_SVC)).to_numpy(float) + \
            p.get("per_prb", 0.0) * tasks.nprb.to_numpy(float)


@register_model("test_static")
class StaticModel(FixedModel):
    """Test only: `FixedModel` with a static power of 10 W (ADR-0004 on-periods)."""

    static_power_w = 10.0


def make_stream(arrivals: list[float], nprb: list[int] | None = None) -> TaskStream:
    """One chain per arrival: task 2j = cpu_processing, 2j+1 = ldpc_decoding depending on it."""
    return synthetic_stream(arrivals, nprb)


def topology(servers: list[list[dict]]) -> Topology:
    """`servers`: per server, executor dicts; model defaults to `test_fixed`."""
    return Topology.from_dict({"servers": [
        {"executors": [{"model": "test_fixed", **e} for e in execs]} for execs in servers]})


class RandomSwitch(SwitchPolicy):
    def __init__(self, n_servers: int, dim: int = 0):
        self.n, self.dim = n_servers, dim

    def act(self, obs):
        msg = Message(self.rng.random(self.dim)) if self.dim else NULL_MESSAGE
        return SwitchAction(int(self.rng.integers(self.n)), msg)


class FixedSwitch(SwitchPolicy):
    def __init__(self, server: int = 0):
        self.server = server

    def act(self, obs):
        return SwitchAction(self.server)


class RandomServer(ServerPolicy):
    def act(self, obs):
        ok = [i for i, e in enumerate(obs.executors) if e.allowed]
        return ServerAction(int(self.rng.choice(ok)))


class FixedServer(ServerPolicy):
    def __init__(self, executor: int = 1):
        self.executor = executor

    def act(self, obs):
        return ServerAction(self.executor)


def run(sim: Simulation, switch: SwitchPolicy, server: ServerPolicy, seed: int = 0,
        log: list | None = None):
    """The ARCHITECTURE §5 loop; optionally logs every DecisionRequest."""
    switch.reset(seed)
    server.reset(seed + 1)
    while (req := sim.next_decision()) is not None:
        if log is not None:
            log.append(req)
        sim.apply((switch if req.tier == "switch" else server).act(req.obs))
    return sim.finalize()


def sim_for(stream, servers, **cfg) -> Simulation:
    return Simulation(stream, topology(servers), SimConfig(**cfg))
