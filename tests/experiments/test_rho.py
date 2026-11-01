"""Offered load rho (ADR-0009): LP against hand-solved cases, class aggregation against the exact LP."""
from __future__ import annotations

import numpy as np
import pytest

from acmad.experiments.rho import job_work, offered_load, rho, rho_peak
from acmad.sim import Topology, load_topology

from simhelpers import make_stream, topology


def _stream(n, T_ms, nprb=None, seed=0):
    rng = np.random.default_rng(seed)
    s = make_stream(list(np.sort(rng.uniform(0, T_ms, n))), nprb)
    s.meta["T_ms"] = T_ms
    return s


def test_single_cpu_is_erlang_per_unit():
    """One server with only a CPU of c cores: rho = lambda * E[S_job] / c."""
    T, n, c = 1000.0, 400, 2
    topo = Topology.from_dict({"servers": [{"executors": [
        {"kind": "cpu", "capacity": c, "model": "synthetic",
         "params": {"dist": "exp", "mean": {"cpu_processing": 0.5, "ldpc_decoding": 1.5}}}]}]})
    r = rho(job_work(_stream(n, T), topo))
    assert r.rho == pytest.approx(n / T * 2.0 / c)


def test_cpu_gpu_split_balances_the_two():
    """cpu 0.5 ms pinned, ldpc 1 ms on either unit (one each): 0.5 + x = 1 - x -> rho = 0.75 lambda."""
    T, n = 1000.0, 300
    r = rho(job_work(_stream(n, T), topology([[{"kind": "cpu"}, {"kind": "gpu"}]])))
    assert r.rho == pytest.approx(0.75 * n / T)
    assert r.utilisation == pytest.approx([0.75 * n / T] * 2)


def test_identical_servers_halve_rho():
    one = [[{"kind": "cpu"}, {"kind": "gpu"}]]
    s = _stream(200, 1000.0)
    assert rho(job_work(s, topology(one * 2))).rho == pytest.approx(rho(job_work(s, topology(one))).rho / 2)


def test_gpu_less_server_runs_everything_on_its_cpu():
    T, n = 1000.0, 100
    r = rho(job_work(_stream(n, T), topology([[{"kind": "cpu", "capacity": 3}]])))
    assert r.rho == pytest.approx(n / T * 1.5 / 3)


def test_classes_match_the_exact_lp():
    """Aggregating jobs by their GPU/CPU work ratio loses almost nothing (ADR-0009 §3)."""
    rng = np.random.default_rng(1)
    n, T = 600, 200.0
    s = _stream(n, T, nprb=list(rng.integers(1, 250, n)), seed=2)
    servers = [[{"kind": "cpu", "capacity": 4, "params": {"per_prb": 0.004}},
                {"kind": "gpu", "capacity": 1, "params": {"svc": {"cpu_processing": 0.5, "ldpc_decoding": 0.2},
                                                          "per_prb": 0.0005}}],
               [{"kind": "cpu", "capacity": 2, "params": {"per_prb": 0.004}}]]
    w = job_work(s, topology(servers))
    exact, agg = rho(w, n_classes=None), rho(w, n_classes=50)
    assert agg.rho >= exact.rho - 1e-9
    assert agg.rho == pytest.approx(exact.rho, rel=0.01)


def test_peak_is_at_least_the_mean_for_bursty_arrivals():
    T = 1000.0
    s = make_stream(list(np.r_[np.linspace(0, 100, 150), np.linspace(100, 1000, 50)]))
    s.meta["T_ms"] = T
    w = job_work(s, topology([[{"kind": "cpu"}, {"kind": "gpu"}]]))
    opt = rho(w)
    assert rho_peak(w, opt) > 2 * opt.rho


@pytest.mark.trace
def test_real_episode_on_the_reference_topology():
    from acmad.traffic import generate, load_config

    cfg = load_config()
    if not cfg.trace_path.exists() or not (cfg.cloudric_dir / "CPU_dataset.csv").exists():
        pytest.skip("real trace or CloudRIC data not available")
    topo = load_topology("configs/topology/example.yaml")
    one = offered_load(generate(cfg, "train", 1, 1.0, 0), topo)
    three = offered_load(generate(cfg, "train", 3, 1.0, 0), topo)
    assert 0 < one["rho"] < three["rho"] <= one["rho_peak"] * 10
    assert max(three["utilisation_at_optimum"]) == pytest.approx(three["rho"], rel=1e-4)
