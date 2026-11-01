"""`ldpc_decoding` on CloudRIC's measured decoders: lookup + resampled joint residual (ADR-0003, ADR-0004).

For each (kind, MCS, nPRB, SNR) the tables hold the median time and median energy of the 9
CloudRIC runs. A draw multiplies both by `exp` of one residual pair `(eps_t, eps_E)` resampled
from the pooled residuals of the kind and of the task's `total_bits` quartile band (the heavy
tail is kept as measured). The energy is split (ADR-0004): the model returns the **dynamic**
part `E - P_static * t` (clipped at 0); `P_static` is the measured power floor (1st
percentile) for `gpu` and 0 for `cpu`.

Tables are built once from `data/cloudric/` and cached in `data/cache/service_models/<hash>/`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from acmad import cache
from acmad.cloudric import DEFAULT_CACHE_DIR, DEFAULT_DIR, read_cloudric
from acmad.nr import N_MCS

VERSION = "1"
DATASET = {"cpu": "CPU", "gpu": "GPU"}         # executor kind -> CloudRIC dataset
N_BANDS = 4                                     # total_bits quartile bands of the residual pools
N_PRB, SNR_MAX = 250, 30
STATIC_FLOOR_Q = 0.01                           # GPU static power = this quantile of the measured power


@dataclass(frozen=True, eq=False)
class LdpcTables:
    """Per kind. `t_ms[m, n-1, snr]`, `e_j[m, n-1, snr]`: medians (NaN = not measured);
    `band_edges`: inner edges of the `total_bits` bands; `res[b]`: (N_b, 2) array of (eps_t, eps_E)."""

    kind: str
    t_ms: np.ndarray
    e_j: np.ndarray
    band_edges: np.ndarray
    res: tuple[np.ndarray, ...]
    static_power_w: float


def build_tables(d: pd.DataFrame, kind: str) -> LdpcTables:
    """Tables of one kind from its CloudRIC dataset (rows with a corrupt energy are dropped)."""
    d = d.dropna(subset=["Decoding_Energy_uJ"]).copy()
    d["snr"] = d.SNR_dB.astype(int)
    key = ["MCS", "nPRB", "snr"]
    g = d.groupby(key)
    med_t = g.Decoding_Time_us.transform("median")
    med_e = g.Decoding_Energy_uJ.transform("median")
    eps = np.c_[np.log(d.Decoding_Time_us / med_t), np.log(d.Decoding_Energy_uJ / med_e)]
    edges = np.quantile(d.Total_Bits, np.linspace(0, 1, N_BANDS + 1)[1:-1])
    band = np.searchsorted(edges, d.Total_Bits.to_numpy(), side="right")
    m = g[["Decoding_Time_us", "Decoding_Energy_uJ"]].median().reset_index()
    t_ms = np.full((N_MCS, N_PRB, SNR_MAX + 1), np.nan)
    e_j = np.full_like(t_ms, np.nan)
    idx = (m.MCS.to_numpy(), m.nPRB.to_numpy() - 1, m.snr.to_numpy())
    t_ms[idx] = m.Decoding_Time_us.to_numpy() * 1e-3
    e_j[idx] = m.Decoding_Energy_uJ.to_numpy() * 1e-6
    static = float(np.quantile(d.Decoding_Power_W, STATIC_FLOOR_Q)) if kind == "gpu" else 0.0
    return LdpcTables(kind, t_ms, e_j, edges, tuple(eps[band == b] for b in range(N_BANDS)), static)


def _save(tabs: dict[str, LdpcTables], d: Path, config: dict) -> None:
    arrays = {}
    for k, t in tabs.items():
        arrays |= {f"{k}_t_ms": t.t_ms, f"{k}_e_j": t.e_j, f"{k}_edges": t.band_edges,
                   f"{k}_static": np.array(t.static_power_w), **{f"{k}_res{b}": r for b, r in enumerate(t.res)}}
    np.savez(d / "tables.npz", **arrays)
    summary = {k: {"static_power_w": t.static_power_w, "band_edges_bits": t.band_edges.tolist(),
                   "residuals_per_band": [len(r) for r in t.res]} for k, t in tabs.items()}
    (d / cache.MANIFEST).write_text(json.dumps({"config": config, "summary": summary}, indent=1))


def _load(d: Path) -> dict[str, LdpcTables]:
    with np.load(d / "tables.npz") as z:
        return {k: LdpcTables(k, z[f"{k}_t_ms"], z[f"{k}_e_j"], z[f"{k}_edges"],
                              tuple(z[f"{k}_res{b}"] for b in range(N_BANDS)), float(z[f"{k}_static"]))
                for k in DATASET}


@lru_cache(maxsize=2)
def load_tables(cloudric_dir: Path = DEFAULT_DIR, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, LdpcTables]:
    """The tables of every kind, from the cache (built from `data/cloudric/` on first use)."""
    config = {"cloudric_sha256": {f: cache.file_sha256(cloudric_dir / f"{f}_dataset.csv", cache_dir)
                                  for f in DATASET.values()},
              "n_bands": N_BANDS, "static_floor_q": STATIC_FLOOR_Q, "version": VERSION}
    d = Path(cache_dir) / "service_models" / cache.key(config)
    if not cache.is_complete(d):
        cr = read_cloudric(cloudric_dir)
        tabs = {k: build_tables(cr[f], k) for k, f in DATASET.items()}
        cache.write_once(d, lambda tmp: _save(tabs, tmp, config))
    return _load(d)


class CloudricLdpcModel:
    """Mixin for the `cpu` / `gpu` models: `ldpc_decoding` from the tables of `KIND`."""

    KIND: str

    @property
    def static_power_w(self) -> float:
        return load_tables()[self.KIND].static_power_w

    def ldpc(self, task, rng: np.random.Generator) -> tuple[float, float]:
        return draw(load_tables()[self.KIND], task.mcs, task.nprb, task.snr_db, task.total_bits, rng)

    def ldpc_mean_ms(self, tasks: pd.DataFrame) -> np.ndarray:
        """E[sigma] of ldpc tasks (rows of a TaskStream frame): t_med * mean(exp(eps_t)) over the
        residual pool of each task's `total_bits` band (ADR-0009 §2)."""
        return mean_time_ms(load_tables()[self.KIND], tasks.mcs.to_numpy(int), tasks.nprb.to_numpy(int),
                            tasks.snr_db.to_numpy(int), tasks.total_bits.to_numpy(int))


def mean_time_ms(tab: LdpcTables, mcs: np.ndarray, nprb: np.ndarray, snr_db: np.ndarray,
                 total_bits: np.ndarray) -> np.ndarray:
    """Vectorised E[sigma] of ldpc tasks: median time times the mean of exp(eps_t) of the band."""
    factor = np.array([np.exp(r[:, 0]).mean() for r in tab.res])
    band = np.searchsorted(tab.band_edges, total_bits, side="right")
    return tab.t_ms[mcs, nprb - 1, snr_db] * factor[band]


def draw(tab: LdpcTables, mcs: int, nprb: int, snr_db: int, total_bits: int,
         rng: np.random.Generator) -> tuple[float, float]:
    """`(service_time_ms, dynamic_energy_j)` of one ldpc task (ADR-0003, ADR-0004)."""
    t0, e0 = tab.t_ms[mcs, nprb - 1, snr_db], tab.e_j[mcs, nprb - 1, snr_db]
    if np.isnan(t0) or np.isnan(e0):
        raise ValueError(f"(MCS {mcs}, nPRB {nprb}, SNR {snr_db}) not measured on {tab.kind} (ADR-0020 should prevent this)")
    pool = tab.res[int(np.searchsorted(tab.band_edges, total_bits, side="right"))]
    eps_t, eps_e = pool[rng.integers(len(pool))]
    t = float(t0 * np.exp(eps_t))
    e = float(e0 * np.exp(eps_e))
    return t, max(e - tab.static_power_w * t * 1e-3, 0.0)
