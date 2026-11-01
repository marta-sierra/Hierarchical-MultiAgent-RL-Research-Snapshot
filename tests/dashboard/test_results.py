"""Results browser backend on a synthetic run folder (no traces needed)."""
import json

import pytest
import yaml

from acmad.dashboard.results import list_runs, resolve, run_detail

METRICS = {"T_ms": 1000.0, "n_jobs": 10, "job_hit_rate": 0.9, "energy_per_job_mj": 5.0, "latency_ms_p99": float("nan"),
           "servers": [{"server": 0, "job_share": 1.0, "utilisation": 0.5}],
           "executors": [{"server": 0, "executor": 0, "kind": "cpu", "capacity": 4, "utilisation": 0.5}]}


def _run(root, name="exp", run="20261023-120000_abc", seeds=(0, 1), done_seeds=(0, 1), summary=False):
    d = root / name / run
    d.mkdir(parents=True)
    cfg = {"name": name, "seeds": list(seeds), "split": "test", "scenarios": [{"n_dus": 1, "s": 1.0, "load_class": None}],
           "methods": [{"name": "a", "switch": {"name": "round_robin"}, "server": {"name": "least_queued"}},
                       {"name": "b", "switch": {"name": "least_loaded"},
                        "server": [{"name": "weighted_score", "alpha": 0.5}, {"name": "all_cpu"}]}]}
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    (d / "git.txt").write_text("deadbeef\n")
    (d / "log.txt").write_text("start\ndone: 4 cells in summary.csv\n")
    for m in ("a", "b"):
        for x in done_seeds:
            c = d / "cells" / m / "n1_s1_any" / f"seed{x}"
            c.mkdir(parents=True)
            met = METRICS | {"job_hit_rate": 0.9 + 0.01 * x + (0.05 if m == "b" else 0)}
            c.joinpath("metrics.json").write_text(json.dumps({"method": m, "scenario": "n1_s1_any", "n_dus": 1, "s": 1.0,
                                                              "load_class": None, "seed": x, "rho": 0.1, "rho_peak": 0.2,
                                                              "wall_s": 1.0, "metrics": met}))
    if summary:
        from acmad.dashboard.results import _cells
        _cells(d).assign(load_class=None).to_csv(d / "summary.csv", index=False)
    return d


def test_list_and_detail(tmp_path):
    _run(tmp_path, summary=True)
    _run(tmp_path, run="20261023-130000_def", seeds=(0, 1, 2), done_seeds=(0,))
    runs = list_runs(tmp_path)["runs"]
    assert [r["run"] for r in runs] == ["20261023-130000_def", "20261023-120000_abc"]     # newest first
    assert [(r["status"], r["n_done"], r["n_cells"]) for r in runs] == [("running", 2, 6), ("done", 4, 4)]
    assert runs[1]["policies"]["b"] == {"switch": "least_loaded", "server": "weighted_score(alpha=0.5) / all_cpu"}
    d = run_detail(tmp_path, "exp/20261023-120000_abc")
    json.dumps(d, allow_nan=False)                                    # NaN became null
    agg = {a["method"]: a for a in d["aggregate"]}
    assert agg["a"]["job_hit_rate"] == pytest.approx(0.905) and agg["a"]["n_seeds"] == 2
    assert agg["b"]["job_hit_rate_ci95"] > 0 and agg["a"]["latency_ms_p99"] is None
    assert d["metrics"][0] == "job_hit_rate" and "s0e0_utilisation" in d["metrics"]


def test_no_escape(tmp_path):
    _run(tmp_path)
    for bad in ("../x", "exp", "/etc"):
        with pytest.raises(FileNotFoundError):
            resolve(tmp_path.resolve(), bad)
