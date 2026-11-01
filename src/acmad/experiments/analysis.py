"""Compare the methods of a run from its `summary.csv` (one row per method x scenario x seed).

Reported only, no ranking (author, 2026-10-22). Across seeds: mean and 95 % t confidence interval.
Paired: every method of a (scenario, seed) saw the same traffic and service noise, so differences to a
baseline are taken seed by seed, which removes the traffic variance from the comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

DEFAULT_METRICS = ("job_hit_rate", "energy_per_job_mj", "latency_ms_p99", "ldpc_on_gpu_share", "power_mean_w")
SCENARIO = ["scenario", "n_dus", "s", "load_class"]


def load_summary(run_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(run_dir) / "summary.csv")
    df["load_class"] = df.load_class.fillna("any")
    return df


def _ci95(x: pd.Series) -> float:
    n = x.notna().sum()
    return float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def table(summary: pd.DataFrame, metrics=DEFAULT_METRICS) -> pd.DataFrame:
    """Per (scenario, method): mean, 95 % CI half-width and number of seeds of each metric, and mean rho."""
    g = summary.groupby(SCENARIO + ["method"], sort=False, dropna=False)
    parts = {"rho": g.rho.mean(), "n_seeds": g.seed.nunique()}
    for m in metrics:
        parts[m] = g[m].mean()
        parts[f"{m}_ci95"] = g[m].agg(_ci95)
    return pd.DataFrame(parts).reset_index()


def paired(summary: pd.DataFrame, baseline: str, metrics=DEFAULT_METRICS) -> pd.DataFrame:
    """Per (scenario, method): mean over seeds of (method - baseline) on the same seed, with its 95 % CI."""
    keys = SCENARIO + ["seed"]
    base = summary[summary.method == baseline].set_index(keys)[list(metrics)]
    rows = []
    for (sc, method), d in summary[summary.method != baseline].groupby(["scenario", "method"], sort=False, dropna=False):
        diff = d.set_index(keys)[list(metrics)] - base.reindex(d.set_index(keys).index)
        row = {"scenario": sc, "method": method, "n_seeds": int(diff.notna().all(axis=1).sum())}
        for m in metrics:
            row[f"d_{m}"] = float(diff[m].mean())
            row[f"d_{m}_ci95"] = _ci95(diff[m])
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(mean: float, ci: float) -> str:
    if np.isnan(mean):
        return "–"
    digits = 4 if abs(mean) < 10 else 2
    return f"{mean:.{digits}f}" + ("" if np.isnan(ci) else f" ± {ci:.{digits}f}")


def report(run_dir: str | Path, metrics=DEFAULT_METRICS, baseline: str | None = None) -> str:
    """Markdown report: one table per metric (rows: scenarios with their rho; columns: methods)."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    s = load_summary(run_dir)
    t = table(s, metrics)
    methods = [m["name"] for m in cfg["methods"]]
    out = [f"# {cfg['name']}: {run_dir.name}", "",
           f"git {(run_dir / 'git.txt').read_text().strip()} · split `{cfg['split']}` · topology "
           f"`{cfg['topology_path']}` · seeds {cfg['seeds']} · {len(s)} cells", "",
           "Mean ± 95 % CI over seeds. ρ: offered load of the scenario (mean over seeds, ADR-0009).", ""]
    for m in metrics:
        out += [f"## {m}", "", "| scenario | ρ | " + " | ".join(methods) + " |",
                "|---|---|" + "---|" * len(methods)]
        for sc, d in t.groupby("scenario", sort=False):
            d = d.set_index("method")
            cells = [_fmt(d.at[x, m], d.at[x, f"{m}_ci95"]) if x in d.index else "–" for x in methods]
            out.append(f"| {sc} | {d.rho.iloc[0]:.4f} | " + " | ".join(cells) + " |")
        out.append("")
    if baseline is not None:
        p = paired(s, baseline, metrics)
        others = [x for x in methods if x != baseline]
        out += [f"## Paired difference to `{baseline}` (same seed)", ""]
        for m in metrics:
            out += [f"### Δ {m}", "", "| scenario | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
            for sc, d in p.groupby("scenario", sort=False):
                d = d.set_index("method")
                cells = [_fmt(d.at[x, f"d_{m}"], d.at[x, f"d_{m}_ci95"]) if x in d.index else "–" for x in others]
                out.append(f"| {sc} | " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out)
