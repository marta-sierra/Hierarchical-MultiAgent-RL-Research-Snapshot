"""Experiment config, seeds, the episode loop and a full (small) run with resume."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from acmad.experiments.config import Method, load_experiment
from acmad.experiments.runner import policy_seed, simulate
from acmad.sim import SimConfig

from simhelpers import make_stream, topology

TWO = [[{"kind": "cpu", "capacity": 2}, {"kind": "gpu", "capacity": 1}], [{"kind": "cpu", "capacity": 1}]]


def _method(**kw):
    return Method.parse({"name": "m", "switch": "round_robin", "server": "least_queued", **kw})


def test_policy_seeds_are_derived_by_name():
    seeds = {policy_seed(0, "switch"), policy_seed(0, "server", 0), policy_seed(0, "server", 1), policy_seed(1, "switch")}
    assert len(seeds) == 4 and policy_seed(0, "server", 1) == policy_seed(0, "server", 1)


def test_simulate_is_deterministic_and_switches_on_required_oracles():
    stream = make_stream([0.0, 0.3, 0.6, 2.0, 2.1])
    a, cfg = simulate(stream, topology(TWO), SimConfig(deadline_ms=10.0), _method(), 4)
    b, _ = simulate(stream, topology(TWO), SimConfig(deadline_ms=10.0), _method(), 4)
    pd.testing.assert_frame_equal(a.to_frame(), b.to_frame())
    assert not cfg.oracle_service
    _, cfg = simulate(stream, topology(TWO), SimConfig(deadline_ms=10.0), _method(server="fastest"), 4)
    assert cfg.oracle_service and not cfg.oracle_cluster


def test_per_server_policies():
    m = Method.parse({"name": "m", "switch": "random", "server": ["all_gpu", {"name": "weighted_score", "alpha": 0.2}]})
    specs = m.server_specs(2)
    assert [s.name for s in specs] == ["all_gpu", "weighted_score"] and specs[1].kwargs == {"alpha": 0.2}
    with pytest.raises(ValueError):
        m.server_specs(3)


def test_load_experiment_grid_and_overrides(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump({
        "name": "t", "seeds": [0, 1], "scenarios": {"n_dus": [1, 2], "s": [1.0], "load_class": [None, "high"]},
        "sim_overrides": {"deadline_ms": 2.0, "links": {"uplink_ms": 0.5}},
        "methods": [{"name": "a", "switch": "random", "server": "random"}]}))
    cfg = load_experiment(p)
    assert [s.id for s in cfg.scenarios] == ["n1_s1_any", "n1_s1_high", "n2_s1_any", "n2_s1_high"]
    assert cfg.sim.deadline_ms == 2.0 and cfg.sim.links.uplink_ms == 0.5 and cfg.sim.links.downlink_ms == 0.1
    assert len(cfg.hash()) == 16


@pytest.mark.trace
def test_smoke_run_and_resume(tmp_path):
    from acmad.experiments.analysis import paired, report, table
    from acmad.experiments.runner import run_experiment
    from acmad.traffic import load_config

    c = load_config()
    if not c.trace_path.exists() or not (c.cloudric_dir / "CPU_dataset.csv").exists():
        pytest.skip("real trace or CloudRIC data not available")
    cfg = load_experiment(Path(__file__).resolve().parents[2] / "configs/experiment/smoke.yaml")
    run = run_experiment(cfg, tmp_path, workers=2)
    s = pd.read_csv(run / "summary.csv")
    assert len(s) == len(cfg.methods) * len(cfg.seeds) and s.job_hit_rate.between(0, 1).all()
    assert {"config.yaml", "git.txt", "seeds.txt", "log.txt", "traffic.csv"} <= {f.name for f in run.iterdir()}
    run_experiment(cfg, tmp_path, workers=2, resume=run)
    assert "resume: 6 cells already done" in (run / "log.txt").read_text()
    assert pd.read_csv(run / "summary.csv").equals(s)
    assert len(table(s)) == len(cfg.methods) and len(paired(s, "rr")) == len(cfg.methods) - 1
    assert "| n1_s1_any |" in report(run, baseline="rr")
