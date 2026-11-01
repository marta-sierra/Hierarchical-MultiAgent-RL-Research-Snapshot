"""Heuristic and service-oracle policies on hand-built observations; the mechanics/brains boundary."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import acmad.policies  # noqa: F401
from acmad.interface import (
    NULL_MESSAGE, ClusterState, ExecutorInfo, ExecutorView, ServerObs, ServerReport, TaskView, available, make,
)
from acmad.interface.observations import JobView, SwitchObs


def _task():
    return TaskView(1, 0, "ldpc_decoding", np.zeros(10), 3.0, 2.0)


def _server_obs(execs, oracle=None):
    return ServerObs(0.0, 0, _task(), tuple(execs), NULL_MESSAGE, oracle_service=oracle)


def _ev(kind, cap, busy=0, queued=0, allowed=True):
    return ExecutorView(kind, cap, busy, queued, 0.0, 0, allowed)


def _switch_obs(in_system, units, since_ack):
    reports = tuple(ServerReport(np.array([[b, 0, 0.0, 0]]), 0.0, 0.0) for b in in_system)
    topo = tuple((ExecutorInfo("cpu", u),) for u in units)
    cluster = ClusterState(0.0, 0.0, np.array(since_ack, float), np.zeros(len(units)))
    return SwitchObs(0.0, JobView(0, 0, 0.0, 3.0, 3.0, (_task(),)), cluster, reports, topo)


def test_registered_names():
    assert set(available("switch")) >= {"random", "round_robin", "least_loaded"}
    assert set(available("server")) >= {"random", "round_robin", "all_cpu", "all_gpu", "least_queued",
                                        "fastest", "cheapest", "weighted_score"}


def test_least_loaded_counts_dispatches_since_the_report_and_capacity():
    p = make("switch", "least_loaded")
    p.reset(0)
    # loads: (4 + 0)/4 = 1, (1 + 2*1)/2 = 1.5, (0 + 0)/1 = 0
    assert p.act(_switch_obs([4, 1, 0], [4, 2, 1], [0, 1, 0])).server == 2
    # loads: 1, 1.5, (0 + 2*3)/1 = 6
    assert p.act(_switch_obs([4, 1, 0], [4, 2, 1], [0, 1, 3])).server == 0


def test_least_loaded_breaks_ties_at_random_reproducibly():
    obs = _switch_obs([0, 0, 0], [1, 1, 1], [0, 0, 0])
    p = make("switch", "least_loaded")
    p.reset(3)
    a = [p.act(obs).server for _ in range(30)]
    p.reset(3)
    assert a == [p.act(obs).server for _ in range(30)] and set(a) == {0, 1, 2}


def test_prefer_kind_and_fallback():
    execs = [_ev("cpu", 16), _ev("gpu", 1)]
    assert make("server", "all_gpu").act(_server_obs(execs)).executor == 1
    assert make("server", "all_cpu").act(_server_obs(execs)).executor == 0
    assert make("server", "all_gpu").act(_server_obs([_ev("cpu", 8)])).executor == 0


def test_least_queued_per_unit_and_allowed():
    execs = [_ev("cpu", 16, busy=16, queued=2), _ev("gpu", 2, busy=1)]
    assert make("server", "least_queued").act(_server_obs(execs)).executor == 1     # 18/16 vs 1/2
    execs = [_ev("cpu", 4, busy=1), _ev("gpu", 1, busy=0, allowed=False)]
    assert make("server", "least_queued").act(_server_obs(execs)).executor == 0


def test_service_oracle_policies():
    execs = [_ev("cpu", 16), _ev("gpu", 1)]
    oracle = ((0.7, 0.0002), (0.07, 0.003))         # cpu: slow, cheap; gpu: fast, costly
    assert make("server", "fastest").act(_server_obs(execs, oracle)).executor == 1
    assert make("server", "cheapest").act(_server_obs(execs, oracle)).executor == 0
    # alpha * ms + (1 - alpha) * mJ: alpha = 1 -> fastest, alpha = 0 -> cheapest
    assert make("server", "weighted_score", alpha=1.0).act(_server_obs(execs, oracle)).executor == 1
    assert make("server", "weighted_score", alpha=0.0).act(_server_obs(execs, oracle)).executor == 0
    assert make("server", "fastest").REQUIRES == {"oracle_service"}
    with pytest.raises(RuntimeError):
        make("server", "fastest").act(_server_obs(execs))
    oracle = ((0.7, 0.0002), None)                  # gpu not allowed
    assert make("server", "fastest").act(_server_obs(execs, oracle)).executor == 0


def test_policies_never_import_the_simulator():
    """ARCHITECTURE §2: brains see only `acmad.interface`."""
    root = Path(acmad.policies.__file__).parent
    for f in root.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.startswith(("acmad.sim", "acmad.traffic")) for n in names), f"{f.name} imports {names}"
