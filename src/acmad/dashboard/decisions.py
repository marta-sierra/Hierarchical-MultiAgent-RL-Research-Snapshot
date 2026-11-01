"""Decisions of one cell (method, scenario, seed) of a run, for the results browser: what each actor
decided for every job / task, next to the task's features (the legacy `decision_scatter`).

- **Switch**: one point per job, coloured by the server it was sent to.
- **Server k**: one point per task its policy decided (ldpc; rule-routed tasks are not decisions,
  ADR-0011), coloured by the executor chosen.

The records come from the cell's `records.parquet` when the run saved them; otherwise the cell is
re-simulated from the run's resolved config (same seeds, so the same episode while the code is the
same as the run's git hash). The task features come from the cell's TaskStream (cached).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from acmad.experiments.config import Method, Scenario
from acmad.experiments.runner import cell_dir, simulate
from acmad.interface.observations import FEATURE_NAMES
from acmad.sim import SimConfig, Topology
from acmad.traffic.api import load_or_generate
from acmad.traffic.config import TrafficConfig

MAX_POINTS = 6000                    # per plot; a random subset beyond that (seeded)
TIME_AXES = ("t_arrival_ms", "latency_ms", "wait_ms")


def _scenario(cfg: dict[str, Any], scenario_id: str) -> Scenario:
    for s in cfg["scenarios"]:
        sc = Scenario(int(s["n_dus"]), float(s["s"]), s.get("load_class"))
        if sc.id == scenario_id:
            return sc
    raise FileNotFoundError(f"no scenario {scenario_id!r} in this run")


@lru_cache(maxsize=8)
def _frames(run_dir: str, method: str, scenario_id: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, str, Topology]:
    d = Path(run_dir)
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    sc = _scenario(cfg, scenario_id)
    spec = next((m for m in cfg["methods"] if m["name"] == method), None)
    if spec is None or int(seed) not in [int(x) for x in cfg["seeds"]]:
        raise FileNotFoundError(f"no cell {method} / {scenario_id} / seed {seed} in this run")
    topo = Topology.from_dict(cfg["topology"])
    stream = load_or_generate(TrafficConfig.from_dict(cfg["traffic"]), cfg["split"], sc.n_dus, sc.s, int(seed),
                              sc.load_class)
    saved = cell_dir(d, method, sc, int(seed)) / "records.parquet"
    if saved.exists():
        rec, source = pd.read_parquet(saved), "records.parquet of the run"
    else:
        r, _ = simulate(stream, topo, SimConfig.from_dict(cfg["sim"]), Method.parse(spec), int(seed))
        rec, source = r.to_frame(), "re-simulated (the run did not save records)"
    feats = stream.tasks.set_index("task_id")[list(FEATURE_NAMES)]
    rec = rec.join(feats, on="task_id")
    rec["latency_ms"] = rec.t_end_ms - rec.t_arrival_ms
    rec["wait_ms"] = rec.t_start_ms - rec.t_ready_ms
    return rec, stream.jobs, source, topo


def _labels(specs) -> list[str]:
    """`cpu`, and `gpu` or `gpu1`, `gpu2`, ... when the server has several GPUs (one executor each)."""
    n = {k: sum(s.kind == k for s in specs) for k in {s.kind for s in specs}}
    seen: dict[str, int] = {}
    out = []
    for s in specs:
        seen[s.kind] = seen.get(s.kind, 0) + 1
        out.append(s.kind if n[s.kind] == 1 else f"{s.kind}{seen[s.kind]}")
    return out


def _sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=seed).sort_index()


def _cols(df: pd.DataFrame, axes: list[str], extra: list[str]) -> dict[str, list]:
    out = {}
    for c in axes + extra:
        v = df[c]
        out[c] = [None if pd.isna(x) else (x.item() if hasattr(x, "item") else x) for x in v.to_numpy()]
    return out


def cell_decisions(run_dir: Path, method: str, scenario_id: str, seed: int, max_points: int = MAX_POINTS) -> dict[str, Any]:
    rec, jobs, source, topo = _frames(str(run_dir), method, scenario_id, int(seed))
    axes = list(FEATURE_NAMES) + list(TIME_AXES)
    # switch: one point per job, the features of its ldpc task (the job's transport block)
    ld = rec[rec.kind == "ldpc_decoding"].copy()
    jobs_out = (rec.groupby("job_id").outcome.agg(lambda o: "met" if (o == "met").all() else "miss"))
    sw = ld.assign(job_outcome=ld.job_id.map(jobs_out)).dropna(subset=["server"])
    sw = _sample(sw, max_points, 0)
    servers = []
    for k, specs in enumerate(topo.servers):
        dec = rec[(rec.server == k) & rec.server_decision_id.notna()]
        n_total = len(dec)
        dec = _sample(dec, max_points, k + 1)
        labels = _labels(specs)
        servers.append({"server": k, "executors": labels, "n_total": n_total,
                        "counts": [int((rec[(rec.server == k) & rec.server_decision_id.notna()].executor == i).sum())
                                   for i in range(len(specs))],
                        "cols": _cols(dec, axes, ["executor", "outcome"])})
    return {"method": method, "scenario": scenario_id, "seed": int(seed), "source": source,
            "axes": axes, "n_servers": topo.n_servers,
            "switch": {"n_total": int(ld.server.notna().sum()),
                       "counts": [int((ld.server == k).sum()) for k in range(topo.n_servers)],
                       "cols": _cols(sw, axes, ["server", "job_outcome"])},
            "servers": servers}
