"""Results browser backend: every run folder under `results/` (ADR-0021 layout), read-only.

A run is `results/<experiment>/<timestamp>_<confighash>/`. Finished runs are read from `summary.csv`;
runs still going (or interrupted) from the `metrics.json` of their done cells, so they show up as
they progress. Aggregation is `acmad.experiments.analysis.table` (mean and 95 % CI over seeds).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from acmad.experiments.analysis import DEFAULT_METRICS, SCENARIO, table
from acmad.objectives.metrics import flat

RUNNING_IF_LOG_AGE_S = 600      # no summary.csv and log touched in the last 10 min: still running
NOT_METRICS = {"method", "scenario", "n_dus", "s", "load_class", "seed", "oracle_cluster", "oracle_service",
               "wall_s", "T_ms", "rho", "rho_peak"}


def _clean(x: Any) -> Any:
    """JSON-safe: NaN / inf -> None, numpy scalars -> Python."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def run_dirs(root: Path) -> list[Path]:
    """Every run folder (one with a `config.yaml`), newest first."""
    return sorted((p.parent for p in root.glob("*/*/config.yaml")), key=lambda p: p.name, reverse=True)


def resolve(root: Path, run_id: str) -> Path:
    """`<experiment>/<run>` -> its folder, refusing anything outside `root` (absolute)."""
    d = (root / run_id).resolve()
    if not d.is_relative_to(root) or not (d / "config.yaml").is_file():
        raise FileNotFoundError(f"no run {run_id!r}")
    return d


def _cells(d: Path) -> pd.DataFrame:
    """One flat row per done cell, like `summary.csv`."""
    if (d / "summary.csv").exists():
        df = pd.read_csv(d / "summary.csv")
    else:
        rows = []
        for f in d.glob("cells/*/*/seed*/metrics.json"):
            c = json.loads(f.read_text())
            rows.append({k: v for k, v in c.items() if k != "metrics"} | flat(c["metrics"]))
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values(["method", "scenario", "seed"])
    if len(df):
        df["load_class"] = df.load_class.fillna("any")
    return df


def _policy(spec: dict[str, Any]) -> str:
    """`{name: weighted_score, alpha: 0.5}` -> `weighted_score(alpha=0.5)`."""
    args = ", ".join(f"{k}={v}" for k, v in spec.items() if k != "name")
    return spec["name"] + (f"({args})" if args else "")


def _policies(method: dict[str, Any]) -> dict[str, str]:
    srv = method["server"]
    return {"switch": _policy(method["switch"]),
            "server": " / ".join(_policy(x) for x in srv) if isinstance(srv, list) else _policy(srv)}


def _status(d: Path, n_done: int, n_expected: int) -> str:
    log = d / "log.txt"
    if (d / "summary.csv").exists():
        last = log.read_text().strip().splitlines()[-1] if log.exists() else ""
        return "failed" if "FAILED" in last else "done"
    if log.exists() and time.time() - log.stat().st_mtime < RUNNING_IF_LOG_AGE_S:
        return "running"
    return "incomplete" if n_done < n_expected else "done"


def _info(d: Path, root: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    methods = [m["name"] for m in cfg.get("methods", [])]
    n_expected = len(methods) * len(cfg.get("scenarios", [])) * len(cfg.get("seeds", []))
    n_done = sum(1 for _ in d.glob("cells/*/*/seed*/metrics.json"))
    git = (d / "git.txt").read_text().strip() if (d / "git.txt").exists() else ""
    stamp = d.name.split("_")[0]
    return {"id": str(d.relative_to(root)), "experiment": cfg.get("name", d.parent.name), "run": d.name,
            "started": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}" if len(stamp) >= 13 else stamp,
            "git": git, "split": cfg.get("split"), "methods": methods,
            "policies": {m["name"]: _policies(m) for m in cfg.get("methods", [])}, "seeds": cfg.get("seeds", []),
            "scenarios": cfg.get("scenarios", []), "n_cells": n_expected, "n_done": n_done,
            "status": _status(d, n_done, n_expected)}


def list_runs(root: Path) -> dict[str, Any]:
    """Every run with its status and, per (method, scenario), rho and the headline metrics (mean over seeds)."""
    root, runs = root.resolve(), []
    for d in run_dirs(root):
        info = _info(d, root)
        df = _cells(d)
        cols = ["n_dus", "s", "rho", "seed"] + [k for k in DEFAULT_METRICS if k in df]
        how = {c: "nunique" if c == "seed" else "first" if c in ("n_dus", "s") else "mean" for c in cols}
        info["headline"] = (df.groupby(["method", "scenario"], sort=False)[cols].agg(how)
                            .rename(columns={"seed": "n_seeds"}).reset_index().to_dict("records") if len(df) else [])
        runs.append(info)
    return _clean({"root": str(root), "runs": runs, "headline": list(DEFAULT_METRICS)})


def run_detail(root: Path, run_id: str) -> dict[str, Any]:
    """Everything the page shows for one run: config, per-cell rows, per (scenario, method) mean ± CI,
    traffic per (scenario, seed), report.md and the end of the log."""
    root = root.resolve()
    d = resolve(root, run_id)
    info = _info(d, root)
    df = _cells(d)
    metrics = [c for c in df.columns if c not in NOT_METRICS and pd.api.types.is_numeric_dtype(df[c])]
    agg = table(df, metrics) if len(df) else pd.DataFrame()
    if len(agg):         # scenarios in config order, methods in config order
        order = {f"{s}": i for i, s in enumerate(df.drop_duplicates("scenario").sort_values(["n_dus", "s"]).scenario)}
        agg = agg.assign(_o=agg.scenario.map(order), _m=agg.method.map({m: i for i, m in enumerate(info["methods"])}))
        agg = agg.sort_values(["_o", "_m"]).drop(columns=["_o", "_m"])
    traffic = pd.read_csv(d / "traffic.csv").fillna({"load_class": "any"}) if (d / "traffic.csv").exists() else pd.DataFrame()
    log = (d / "log.txt").read_text().splitlines() if (d / "log.txt").exists() else []
    text = (d / "config.yaml").read_text()
    return _clean(info | {
        "config": text, "config_dict": yaml.safe_load(text),
        "metrics": [m for m in DEFAULT_METRICS if m in metrics] + [m for m in metrics if m not in DEFAULT_METRICS],
        "scenario_keys": SCENARIO,
        "aggregate": agg.to_dict("records"), "cells": df.to_dict("records"),
        "traffic": traffic.to_dict("records"),
        "report": (d / "report.md").read_text() if (d / "report.md").exists() else None,
        "log_tail": log[-40:]})
