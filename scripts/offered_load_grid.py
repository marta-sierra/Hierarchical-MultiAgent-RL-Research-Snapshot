"""Offered traffic in Erlangs over the benchmark grid (n_dus x s), as heatmaps for the thesis.

    .venv/bin/python scripts/offered_load_grid.py [--experiment configs/experiment/benchmark.yaml]
        [--out docs/summaries_tfg/figures]

Per (n_dus, s) and seed of the experiment, A = sum of the mean service times of the episode's tasks / T
(Erlangs = units busy on average), with each model's closed-form mean (ADR-0009 §2):
- `erl_cpu_only`: every task on CPU cores (cpu_processing + ldpc);
- `erl_gpu_ldpc_cpu` / `erl_gpu_ldpc_gpu`: cpu_processing on CPU cores, ldpc on GPUs (CPU / GPU Erlangs);
and the fluid-optimal utilisation rho of the reference topology (ADR-0009). Writes
`offered_load_grid.csv` (one row per scenario, mean over seeds) and `offered_load_grid.png`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

from acmad.experiments.config import REPO, load_experiment  # noqa: E402
from acmad.experiments.rho import offered_load  # noqa: E402
from acmad.sim.models import ExecutorSpec, make_model  # noqa: E402
from acmad.traffic import load_config  # noqa: E402
from acmad.traffic.api import load_or_generate  # noqa: E402

# Sequential blue ramp (dataviz palette, steps 100 -> 700): one hue, light = low, dark = high.
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
INK, MUTED = "#1f1f1e", "#6b6a64"


def erlangs(stream) -> dict[str, float]:
    tasks, T = stream.tasks, float(stream.meta["T_ms"])
    cpu = make_model("cpu").mean_service_ms(tasks, ExecutorSpec("cpu"))
    gpu = make_model("gpu").mean_service_ms(tasks, ExecutorSpec("gpu"))
    ld = tasks.kind.to_numpy(str) == "ldpc_decoding"
    return {"erl_cpu_only": np.nansum(cpu) / T,
            "erl_gpu_ldpc_cpu": np.nansum(cpu[~ld]) / T, "erl_gpu_ldpc_gpu": np.nansum(gpu[ld]) / T}


def heatmap(ax, grid: pd.DataFrame, title: str, fmt: str, norm) -> None:
    im = ax.imshow(grid.to_numpy(), cmap=BLUES, norm=norm, aspect="auto", origin="lower")
    ax.set_xticks(range(grid.shape[1]), [f"{c:g}" for c in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [str(i) for i in grid.index])
    ax.set_xlabel("s (bits multiplier)", fontsize=8, color=INK)
    ax.set_ylabel("n_dus (cells)", fontsize=8, color=INK)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.iat[i, j]
            dark = norm(v) > 0.55
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if dark else INK)
    return im


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="configs/experiment/benchmark.yaml")
    ap.add_argument("--out", default=str(REPO / "docs/summaries_tfg/figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exp = load_experiment(a.experiment)
    cfg = load_config(exp.traffic_path)
    rows = []
    for sc in exp.scenarios:
        for x in exp.seeds:
            st = load_or_generate(cfg, exp.split, sc.n_dus, sc.s, x, sc.load_class)
            ol = offered_load(st, exp.topology)
            rows.append({"n_dus": sc.n_dus, "s": sc.s, "seed": x, "jobs_per_ms": len(st.jobs) / st.meta["T_ms"],
                         **erlangs(st), "rho": ol["rho"], "rho_peak": ol["rho_peak"]})
        print(f"n_dus {sc.n_dus} s {sc.s:g}", flush=True)
    df = pd.DataFrame(rows).groupby(["n_dus", "s"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    df.to_csv(out / "offered_load_grid.csv", index=False, float_format="%.5g")

    cpu = df.pivot(index="n_dus", columns="s", values="erl_cpu_only")
    rho = df.pivot(index="n_dus", columns="s", values="rho")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    heatmap(axes[0], cpu, "Offered traffic, all on CPU cores [Erlangs]", ".2f",
            LogNorm(vmin=cpu.min().min(), vmax=cpu.max().max()))
    heatmap(axes[1], rho, "Fluid-optimal utilisation ρ of the reference topology", ".3f",
            LogNorm(vmin=rho.min().min(), vmax=rho.max().max()))
    fig.text(0.01, 0.01, f"Mean over seeds {list(exp.seeds)}, {exp.split} split. Erlangs = mean busy units; "
             "the reference topology has 28 CPU cores and 3 GPUs. Colour scales are logarithmic.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "offered_load_grid.png", dpi=200)
    plt.close(fig)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
