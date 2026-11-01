"""Service-time distributions sigma(e, i) over the benchmark knobs (no queueing): ldpc on CPU / GPU and
the whole job, cpu_processing (always on the CPU, ADR-0011) + ldpc on CPU or GPU.

    .venv/bin/python scripts/service_time_distributions.py [--out docs/summaries_tfg/figures]

Samples the simulator's own models (noise included) on `n_jobs` random jobs of each (n_dus, s) of
configs/experiment/benchmark.yaml (test split, seed 0). Writes `service_time_quantiles.csv` (one row
per scenario and quantity) and three figures (histograms of ldpc and of the job, CCDF of the job with
candidate deadlines). Figures pool the `n_dus` values: the per-task distribution depends on `s` only.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from acmad.experiments.config import REPO, load_experiment  # noqa: E402
from acmad.sim.models import ExecutorSpec, TaskFeatures, make_model  # noqa: E402
from acmad.traffic import load_config  # noqa: E402
from acmad.traffic.api import load_or_generate  # noqa: E402

CPU_COLOR, GPU_COLOR = "#2a78d6", "#eb6834"      # categorical slots 1-2, validated (dataviz palette)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
DEADLINES_MS = (1.0, 1.5, 2.0, 3.0)              # candidates compared (ADR-0013: D = 1 ms)
FEATURES = [f.name for f in fields(TaskFeatures)]


def sample(stream, n_jobs: int, rng: np.random.Generator) -> pd.DataFrame:
    """One row per sampled job: cp (cpu_processing on CPU), ld_cpu, ld_gpu, job_cpu, job_gpu [ms]."""
    cpu, gpu = make_model("cpu"), make_model("gpu")
    scpu, sgpu = ExecutorSpec("cpu"), ExecutorSpec("gpu")
    ids = stream.jobs.job_id.to_numpy()
    pick = rng.choice(ids, size=min(n_jobs, len(ids)), replace=False)
    per: dict[int, dict[str, float]] = {}
    for r in stream.tasks[stream.tasks.job_id.isin(pick)].itertuples(index=False):
        t = TaskFeatures(**{f: getattr(r, f) for f in FEATURES})
        d = per.setdefault(r.job_id, {})
        if r.kind == "cpu_processing":
            d["cp"] = cpu(t, scpu, rng)[0]
        else:
            d["ld_cpu"], d["ld_gpu"] = cpu(t, scpu, rng)[0], gpu(t, sgpu, rng)[0]
    df = pd.DataFrame(per.values()).dropna()
    return df.assign(job_cpu=df.cp + df.ld_cpu, job_gpu=df.cp + df.ld_gpu)


def quantiles(df: pd.DataFrame) -> list[dict]:
    rows = []
    for col in ("cp", "ld_cpu", "ld_gpu", "job_cpu", "job_gpu"):
        x = df[col].to_numpy()
        rows.append({"what": col, "mean_ms": x.mean(),
                     **{f"p{q}_ms": np.percentile(x, q) for q in (50, 90, 99, 99.9)}, "max_ms": x.max(),
                     **{f"over_{D}ms": (x > D).mean() for D in DEADLINES_MS}})
    return rows


def _style(ax) -> None:
    ax.set_xscale("log")
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _deadlines(ax, label: bool) -> None:
    for D in DEADLINES_MS:
        ax.axvline(D, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        if label:
            ax.text(D, 1.0, f"{D:g}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=7, color=MUTED)


def histograms(pooled: dict[float, pd.DataFrame], cpu_col: str, gpu_col: str, title: str, path: Path) -> None:
    bins = np.logspace(np.log10(0.01), np.log10(7.0), 45)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey=True)
    for ax, (s, df) in zip(axes.flat, pooled.items()):
        for col, color, name in ((cpu_col, CPU_COLOR, "ldpc on CPU"), (gpu_col, GPU_COLOR, "ldpc on GPU")):
            w = np.full(len(df), 100.0 / len(df))
            ax.hist(df[col], bins=bins, weights=w, histtype="step", linewidth=1.6, color=color, label=name)
        _style(ax)
        _deadlines(ax, label=True)
        ax.set_title(f"s = {s:g}", loc="left", fontsize=9, color=INK, pad=10)
    for ax in axes[1]:
        ax.set_xlabel("service time [ms] (log scale)", fontsize=8, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("% of jobs per bin", fontsize=8, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=10, color=INK)
    fig.text(0.01, 0.005, "Dashed lines: candidate deadlines D [ms]. No queueing. Test split, seed 0, "
             "all n_dus of the benchmark pooled.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def ccdf(pooled: dict[float, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey=True)
    for ax, (s, df) in zip(axes.flat, pooled.items()):
        for col, color, name in (("job_cpu", CPU_COLOR, "ldpc on CPU"), ("job_gpu", GPU_COLOR, "ldpc on GPU")):
            x = np.sort(df[col].to_numpy())
            p = 1.0 - np.arange(1, len(x) + 1) / len(x)
            ax.plot(x[:-1], p[:-1], color=color, linewidth=2, label=name)
        _style(ax)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.2)
        _deadlines(ax, label=True)
        ax.set_title(f"s = {s:g}", loc="left", fontsize=9, color=INK, pad=10)
    for ax in axes[1]:
        ax.set_xlabel("job service time [ms] (log scale)", fontsize=8, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(service time > t)", fontsize=8, color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Job service time (cpu_processing + ldpc): fraction of jobs over t", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.text(0.01, 0.005, "Read at a dashed line: share of jobs that miss deadline D from service time alone "
             "(no queueing). Test split, seed 0, all n_dus pooled.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="configs/experiment/benchmark.yaml")
    ap.add_argument("--n-jobs", type=int, default=20000, help="jobs sampled per (n_dus, s)")
    ap.add_argument("--out", default=str(REPO / "docs/summaries_tfg/figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exp = load_experiment(a.experiment)
    cfg = load_config(exp.traffic_path)
    rows, by_s = [], {}
    for sc in exp.scenarios:
        stream = load_or_generate(cfg, exp.split, sc.n_dus, sc.s, 0, sc.load_class)
        df = sample(stream, a.n_jobs, np.random.default_rng(12345))
        rows += [{"n_dus": sc.n_dus, "s": sc.s, "n_jobs_stream": len(stream.jobs), **r} for r in quantiles(df)]
        by_s.setdefault(sc.s, []).append(df)
        print(f"n_dus {sc.n_dus} s {sc.s:g}: {len(stream.jobs)} jobs", flush=True)
    pd.DataFrame(rows).to_csv(out / "service_time_quantiles.csv", index=False, float_format="%.5g")
    pooled = {s: pd.concat(v, ignore_index=True) for s, v in sorted(by_s.items())}
    histograms(pooled, "ld_cpu", "ld_gpu", "ldpc decoding service time", out / "service_time_ldpc_hist.png")
    histograms(pooled, "job_cpu", "job_gpu", "Job service time: cpu_processing (CPU) + ldpc",
               out / "service_time_job_hist.png")
    ccdf(pooled, out / "service_time_job_ccdf.png")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
