"""Static vs dynamic energy (ADR-0004), figures for the thesis.

    .venv/bin/python scripts/energy_static_vs_dynamic.py [--n-dus 4] [--s 2] [--seed 0]
        [--out docs/summaries_tfg/figures]

1. `energy_cloudric_power.png`: distribution of the decoding power measured by CloudRIC (GPU, CPU), with
   the static power the simulator uses (GPU: 1st percentile of the measured power; CPU: 0).
2. `energy_random_timeline_n<n_dus>_s<s>.png`: one episode of random assignment (random switch, random server) on the
   reference topology: energy per 100 ms bin, static at the bottom and dynamic stacked on top, CPU and
   GPU in separate panels. Dynamic energy of a task is spread uniformly over its service time; static
   energy is P_static times the on-time of each on-period inside the bin.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from acmad.experiments.config import REPO, Method, load_experiment  # noqa: E402
from acmad.experiments.runner import simulate  # noqa: E402
from acmad.sim.models.cloudric_ldpc import load_tables  # noqa: E402
from acmad.traffic import load_config  # noqa: E402
from acmad.traffic.api import load_or_generate  # noqa: E402

STATIC, DYNAMIC = "#9a9890", "#2a78d6"            # neutral for the floor, categorical slot 1 for the work
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
BIN_MS = 100.0


def _style(ax) -> None:
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def cloudric_power(path: Path) -> dict[str, float]:
    tabs = load_tables()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    shares = {}
    for ax, kind, fname in ((axes[0], "gpu", "GPU_dataset.csv"), (axes[1], "cpu", "CPU_dataset.csv")):
        d = pd.read_csv(REPO / "data/cloudric" / fname, low_memory=False, usecols=["Decoding_Power_W", "Decoding_Time_us",
                                                                 "Decoding_Energy_uJ"])
        d = d.apply(pd.to_numeric, errors="coerce").dropna()      # a few corrupt CPU rows (as in the model tables)
        p, ps = d.Decoding_Power_W.to_numpy(), tabs[kind].static_power_w
        lo, hi = np.percentile(p, [0.05, 99.95])
        ax.hist(p, bins=np.linspace(min(lo, ps), hi, 80), weights=np.full(len(p), 100.0 / len(p)),
                color=DYNAMIC, alpha=0.85)
        ax.axvline(ps, color=INK, linewidth=1.4, linestyle=(0, (4, 3)))
        ax.text(ps, 1.0, f" static = {ps:.1f} W", transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=8, color=INK)
        shares[kind] = float(np.sum(ps * d.Decoding_Time_us * 1e-6) / np.sum(d.Decoding_Energy_uJ * 1e-6))
        _style(ax)
        ax.set_title(f"{kind.upper()} ({len(p):,} decodes): static share of the energy {shares[kind]:.0%}",
                     loc="left", fontsize=9, color=INK)
        ax.set_xlabel("measured decoding power [W]", fontsize=8, color=INK)
    axes[0].set_ylabel("% of decodes per bin", fontsize=8, color=INK)
    fig.text(0.01, 0.01, "CloudRIC measurements (data/cloudric). GPU: board power (nvidia-smi); CPU: power "
             "attributed to the decoding process (PowerTOP).", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return shares


def _spread(starts: np.ndarray, ends: np.ndarray, energy: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Energy of each interval spread uniformly over it, summed per bin."""
    out = np.zeros(len(edges) - 1)
    for a, b, e in zip(starts, ends, energy):
        if not e > 0:
            continue
        if b <= a:
            out[min(np.searchsorted(edges, a, side="right") - 1, len(out) - 1)] += e
            continue
        lo, hi = np.maximum(edges[:-1], a), np.minimum(edges[1:], b)
        out += np.clip(hi - lo, 0, None) / (b - a) * e
    return out


def random_timeline(n_dus: int, s: float, seed: int, path: Path) -> dict[str, dict[str, float]]:
    exp = load_experiment(REPO / "configs/experiment/benchmark.yaml")
    stream = load_or_generate(load_config(exp.traffic_path), exp.split, n_dus, s, seed, None)
    method = Method.parse({"name": "random", "switch": "random", "server": "random"})
    records, _ = simulate(stream, exp.topology, exp.sim, method, seed)
    tasks, periods = records.to_frame(), records.on_periods_frame()
    kind_of = {(k, e): spec.kind for k, srv in enumerate(exp.topology.servers) for e, spec in enumerate(srv)}
    tasks = tasks[tasks.t_start_ms.notna()].copy()
    tasks["ex_kind"] = [kind_of[(int(k), int(e))] for k, e in zip(tasks.server, tasks.executor)]
    T = float(stream.meta["T_ms"])
    end = max(T, float(tasks.t_end_ms.max()), float(periods.t_off_ms.max()))
    edges = np.arange(0.0, end + BIN_MS, BIN_MS)
    mid_s = (edges[:-1] + BIN_MS / 2) / 1e3
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True)
    totals = {}
    for ax, kind in zip(axes, ("cpu", "gpu")):
        t = tasks[tasks.ex_kind == kind]
        p = periods[periods.kind == kind]
        dyn = _spread(t.t_start_ms.to_numpy(float), t.t_end_ms.to_numpy(float), t.energy_dynamic_j.to_numpy(float), edges)
        sta = _spread(p.t_on_ms.to_numpy(float), p.t_off_ms.to_numpy(float), p.static_j.to_numpy(float), edges)
        totals[kind] = {"static_j": float(sta.sum()), "dynamic_j": float(dyn.sum()), "n_tasks": int(len(t))}
        ax.bar(mid_s, sta, width=BIN_MS / 1e3, color=STATIC, label="static", linewidth=0)
        ax.bar(mid_s, dyn, width=BIN_MS / 1e3, bottom=sta, color=DYNAMIC, label="dynamic", linewidth=0)
        _style(ax)
        tot = sta.sum() + dyn.sum()
        share = f"{sta.sum() / tot:.1%} static" if tot > 0 else "no energy"
        ax.set_title(f"{kind.upper()}: static {sta.sum():.1f} J + dynamic {dyn.sum():.2f} J ({share}), "
                     f"{len(t):,} tasks", loc="left", fontsize=9, color=INK)
        ax.set_ylabel(f"energy per {BIN_MS:g} ms [J]", fontsize=8, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    axes[1].set_xlabel("time in the episode [s]", fontsize=8, color=INK)
    fig.text(0.01, 0.005, f"Random switch + random server, reference topology (28 CPU cores, 3 GPUs, each "
             f"its own executor), n_dus = {n_dus}, s = {s:g}, seed {seed}, test split, D = {exp.sim.deadline_ms:g} ms, "
             f"T_off = 10 s.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-dus", type=int, default=4)
    ap.add_argument("--s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "docs/summaries_tfg/figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("static share of CloudRIC energy:", cloudric_power(out / "energy_cloudric_power.png"))
    print("random episode:", random_timeline(a.n_dus, a.s, a.seed, out / f"energy_random_timeline_n{a.n_dus}_s{a.s:g}.png"))


if __name__ == "__main__":
    main()
