"""Episode metrics on hand-checkable runs (objectives/metrics.py)."""
from __future__ import annotations

import numpy as np
import pytest

from acmad.objectives.metrics import episode_metrics, flat
from acmad.sim import SimConfig, Simulation

from simhelpers import FixedServer, FixedSwitch, make_stream, run, topology

SERVERS = [[{"kind": "cpu", "capacity": 2}, {"kind": "gpu", "capacity": 1}],
           [{"kind": "cpu", "capacity": 1}]]


def _metrics(arrivals, deadline_ms, server=0, executor=1, T_ms=10.0):
    topo = topology(SERVERS)
    rec = run(Simulation(make_stream(arrivals), topo, SimConfig(deadline_ms=deadline_ms)),
              FixedSwitch(server), FixedServer(executor))
    return episode_metrics(rec.to_frame(), rec.on_periods_frame(), topo, T_ms)


def test_all_met():
    # cpu 0.5 ms, ldpc 1.0 ms on the gpu; FixedModel energy = power_w (1) * time
    m = _metrics([0.0, 2.0], deadline_ms=5.0)
    assert m["n_jobs"] == 2 and m["n_tasks"] == 4
    assert m["job_hit_rate"] == 1.0 and m["task_met_rate"] == 1.0
    assert m["latency_ms_mean"] == pytest.approx(1.5) and m["latency_ms_max"] == pytest.approx(1.5)
    assert m["ldpc_wait_ms_max"] == 0.0
    assert m["energy_dynamic_j"] == pytest.approx(2 * (0.5 + 1.0))
    assert m["ldpc_on_gpu_share"] == 1.0
    cpu, gpu, other = m["executors"]
    assert cpu["n_tasks"] == 2 and cpu["utilisation"] == pytest.approx(2 * 0.5 / (2 * 10.0))
    assert gpu["utilisation"] == pytest.approx(2 * 1.0 / 10.0) and other["n_tasks"] == 0
    assert [s["job_share"] for s in m["servers"]] == [1.0, 0.0]
    assert m["servers"][0]["utilisation"] == pytest.approx((1.0 + 2.0) / (3 * 10.0))


def test_queue_causes_misses_and_drops():
    # three jobs at once on one gpu: ldpc ready at 0.5 for all; ends 1.5, 2.5 (> D = 2: aborted at 2.0), third dropped
    m = _metrics([0.0, 0.0, 0.0], deadline_ms=2.0)
    assert m["job_hit_rate"] == pytest.approx(1 / 3)
    assert m["task_missed_rate"] == pytest.approx(1 / 6) and m["task_dropped_rate"] == pytest.approx(1 / 6)
    assert m["ldpc_wait_ms_max"] == pytest.approx(1.0)


def test_flat_row():
    row = flat(_metrics([0.0], deadline_ms=5.0))
    assert "s0e1_utilisation" in row and "s1_job_share" in row and "executors" not in row
    assert all(np.isscalar(v) for v in row.values())
