"""Run an experiment: every (method, scenario, seed) cell is one episode (ARCHITECTURE §5, §7).

Results go to `results/<name>/<timestamp>_<confighash>/`:

    config.yaml   resolved config (topology, simulator, traffic, methods, scenarios, seeds)
    git.txt       commit (+ "-dirty")          seeds.txt   how every seed is derived (ADR-0002)
    log.txt       progress                     traffic.csv one row per (scenario, seed): size, rho, rho_peak
    summary.csv   one flat row per cell: method, scenario, seed, rho, metrics
    cells/<method>/<scenario>/seed<x>/metrics.json  (+ records.parquet, on_periods.parquet)

Every method of a (scenario, seed) gets the same TaskStream (cached by its key) and the same service
noise (keyed by (x; server, executor, task), ADR-0002), so their differences come from the decisions only.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import acmad.policies  # noqa: F401  (register every policy)
from acmad.experiments.config import REPO, ExperimentConfig, Method, Scenario
from acmad.experiments.rho import offered_load
from acmad.interface import make
from acmad.objectives.metrics import episode_metrics, flat
from acmad.sim import Records, SimConfig, Simulation, Topology, check_records
from acmad.traffic import load_config
from acmad.traffic.api import git_hash, load_or_generate, stream_key
from acmad.traffic.stream import TaskStream

POLICY_STREAM = 1          # spawn-key namespace of the policy rngs (the executors use 0, ADR-0002)
TIERS = {"switch": 0, "server": 1}
SEEDS_TXT = """experiment seed x, for each (scenario, x):
  traffic:  generate(split, n_dus, s, seed=x, load_class)  (the generator spawns its own streams from x)
  service:  Simulation(seed=x): rng of task i on executor e of server k = SeedSequence(x, spawn_key=(0, k, e, i))
  policies: reset(int(SeedSequence(x, spawn_key=(1, tier, k)).generate_state(1)[0])), tier 0 = switch, 1 = server k
"""


def policy_seed(x: int, tier: str, k: int = 0) -> int:
    """Seed of a policy's rng, derived from the experiment seed by name (ADR-0002)."""
    return int(np.random.SeedSequence(x, spawn_key=(POLICY_STREAM, TIERS[tier], k)).generate_state(1)[0])


def simulate(stream: TaskStream, topology: Topology, sim_cfg: SimConfig, method: Method, x: int) -> tuple[Records, SimConfig]:
    """One episode with `method`'s policies (the ARCHITECTURE §5 loop). The oracle modes the policies
    declare in `REQUIRES` are switched on; returns the records and the simulator config used."""
    switch = make("switch", method.switch.name, **method.switch.kwargs)
    servers = [make("server", p.name, **p.kwargs) for p in method.server_specs(topology.n_servers)]
    switch.reset(policy_seed(x, "switch"))
    for k, p in enumerate(servers):
        p.reset(policy_seed(x, "server", k))
    req = set().union(switch.REQUIRES, *(p.REQUIRES for p in servers))
    cfg = replace(sim_cfg, oracle_cluster=sim_cfg.oracle_cluster or "oracle_cluster" in req,
                  oracle_service=sim_cfg.oracle_service or "oracle_service" in req)
    sim = Simulation(stream, topology, cfg, seed=x)
    while (r := sim.next_decision()) is not None:
        sim.apply((switch if r.tier == "switch" else servers[r.server]).act(r.obs))
    return sim.finalize(), cfg


def _stream(cfg: ExperimentConfig, sc: Scenario, x: int) -> TaskStream:
    return load_or_generate(load_config(cfg.traffic_path), cfg.split, sc.n_dus, sc.s, x, sc.load_class)


def _traffic_job(cfg: ExperimentConfig, sc: Scenario, x: int) -> dict[str, Any]:
    """Generate (or read) the stream of a (scenario, seed) and compute its offered load."""
    st = _stream(cfg, sc, x)
    return {"scenario": sc.id, **vars(sc), "seed": x,
            "stream_key": stream_key(load_config(cfg.traffic_path), cfg.split, sc.n_dus, sc.s, x, sc.load_class),
            "n_jobs": len(st.jobs), "T_ms": st.meta["T_ms"], "jobs_per_ms": len(st.jobs) / st.meta["T_ms"],
            **offered_load(st, cfg.topology)}


def cell_dir(run_dir: Path, method: str, sc: Scenario, x: int) -> Path:
    return run_dir / "cells" / method / sc.id / f"seed{x}"


def _cell_job(cfg: ExperimentConfig, run_dir: Path, method: Method, sc: Scenario, x: int,
              load: dict[str, Any]) -> dict[str, Any]:
    """Run one cell and write its files; returns its flat summary row."""
    d = cell_dir(run_dir, method.name, sc, x)
    t0 = time.perf_counter()
    stream = _stream(cfg, sc, x)
    rec, used = simulate(stream, cfg.topology, cfg.sim, method, x)
    tasks, periods = rec.to_frame(), rec.on_periods_frame()
    m = episode_metrics(tasks, periods, cfg.topology, stream.meta["T_ms"])
    if cfg.check_invariants:
        bad = {c.name: c.violations[:5] for c in check_records(tasks, stream, cfg.topology, used.deadline_ms) if not c.ok}
        if bad:
            raise RuntimeError(f"invariants violated in {d}: {bad}")
    d.mkdir(parents=True, exist_ok=True)
    if cfg.save_records:
        tasks.to_parquet(d / "records.parquet", index=False)
        periods.to_parquet(d / "on_periods.parquet", index=False)
    out = {"method": method.name, "scenario": sc.id, **vars(sc), "seed": x,
           "rho": load["rho"], "rho_peak": load["rho_peak"],
           "oracle_cluster": used.oracle_cluster, "oracle_service": used.oracle_service,
           "wall_s": time.perf_counter() - t0, "metrics": m}
    (d / "metrics.json.tmp").write_text(json.dumps(out, indent=1))
    os.replace(d / "metrics.json.tmp", d / "metrics.json")       # written last: marks the cell done
    return _row(out)


def _row(cell: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cell.items() if k != "metrics"} | flat(cell["metrics"])


class _Log:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def new_run_dir(cfg: ExperimentConfig, out_root: Path) -> Path:
    d = out_root / cfg.name / f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.hash()}"
    d.mkdir(parents=True)
    return d


def run_experiment(cfg: ExperimentConfig, out_root: str | Path = REPO / "results", workers: int | None = None,
                   resume: str | Path | None = None) -> Path:
    """Run every cell (in parallel over `workers` processes) and write the run folder; returns it.

    `resume`: an existing run folder of the same config; cells with a `metrics.json` are kept.
    """
    if resume is not None:
        run_dir = Path(resume)
        saved = yaml.safe_load((run_dir / "config.yaml").read_text())
        if saved != json.loads(json.dumps(cfg.resolved())):
            raise ValueError(f"{run_dir} was run with a different config")
    else:
        run_dir = new_run_dir(cfg, Path(out_root))
        (run_dir / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(cfg.resolved())), sort_keys=False))
        (run_dir / "git.txt").write_text(f"{git_hash()}\n")
        (run_dir / "seeds.txt").write_text(SEEDS_TXT)
    log = _Log(run_dir / "log.txt")
    workers = workers or os.cpu_count() or 1
    cells = [(m, sc, x) for sc in cfg.scenarios for x in cfg.seeds for m in cfg.methods]
    log(f"run {run_dir.name}: {len(cfg.methods)} methods x {len(cfg.scenarios)} scenarios x {len(cfg.seeds)} seeds "
        f"= {len(cells)} cells, {workers} workers, git {git_hash()}")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_traffic_job, cfg, sc, x): (sc, x) for sc in cfg.scenarios for x in cfg.seeds}
        loads = {}
        for f in as_completed(futs):
            sc, x = futs[f]
            loads[(sc.id, x)] = f.result()
            log(f"traffic {sc.id} seed {x}: {loads[(sc.id, x)]['n_jobs']} jobs, rho {loads[(sc.id, x)]['rho']:.4f}")
        pd.DataFrame([loads[(sc.id, x)] for sc in cfg.scenarios for x in cfg.seeds]).to_csv(run_dir / "traffic.csv", index=False)

        rows, todo = [], []
        for m, sc, x in cells:
            done = cell_dir(run_dir, m.name, sc, x) / "metrics.json"
            if done.exists():
                rows.append(_row(json.loads(done.read_text())))
            else:
                todo.append((m, sc, x))
        if rows:
            log(f"resume: {len(rows)} cells already done")
        futs = {pool.submit(_cell_job, cfg, run_dir, m, sc, x, loads[(sc.id, x)]): (m, sc, x) for m, sc, x in todo}
        failed = 0
        for i, f in enumerate(as_completed(futs), 1):
            m, sc, x = futs[f]
            try:
                r = f.result()
            except Exception as err:            # keep the other cells running; rerun with --resume
                failed += 1
                log(f"[{i}/{len(todo)}] FAILED {m.name} {sc.id} seed {x}: {type(err).__name__}: {err}")
                continue
            rows.append(r)
            log(f"[{i}/{len(todo)}] {m.name} {sc.id} seed {x}: h = {r['job_hit_rate']:.4f}, "
                f"E = {r['energy_per_job_mj']:.3f} mJ/job, {r['wall_s']:.0f} s")

    keys = ["method", "scenario", "seed"]
    pd.DataFrame(rows).sort_values(keys).to_csv(run_dir / "summary.csv", index=False)
    log(f"done: {len(rows)} cells in summary.csv" + (f", {failed} FAILED (rerun with --resume)" if failed else ""))
    if failed:
        raise RuntimeError(f"{failed} cells failed; see {run_dir / 'log.txt'}")
    return run_dir
