"""Stage R: radio tables. TBS and code-block tables (`acmad.nr`), the SNRs CloudRIC measured per (MCS, nPRB) on both decoders (ADR-0020), and the
option-B LTE -> NR MCS map (ADR-0018).

docs/TRAFFIC.md §6-§8. Cached once in `data/cache/radio_tables/<hash>/`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from acmad import cache
from acmad.cloudric import CLOUDRIC_FILES, read_cloudric
from acmad.nr import N_MCS, NR_EFF, cb_table, tbs_table

from .config import GENERATOR_VERSION, TrafficConfig
from .traces import Prep, ensure_prep

N_LTE_MCS = 29                                                  # LTE MCS 0-28 (29-31 dropped in A5)
LTE_CEILING_MCS = 28


@dataclass(frozen=True, eq=False)
class Radio:
    """Stage R output.

    `tbs[m, n-1]`: TBS of MCS m on n PRBs; `cb[m, n-1] = (base graph, n_cb, cb_len)`;
    `snr_min[m, n-1]`: lowest SNR (dB) at which (MCS m, n PRBs) is measured on both CloudRIC
    decoders; every integer SNR from it to `snr_max` is measured too, except the triples
    `(m, n, snr)` listed in `snr_holes` (ADR-0020);
    `map_lo / map_hi / map_p[lte]`: option-B map, NR MCS hi with probability p, else lo.
    """

    key: str
    tbs: np.ndarray
    cb: np.ndarray
    snr_min: np.ndarray
    snr_max: int
    snr_holes: np.ndarray
    map_lo: np.ndarray
    map_hi: np.ndarray
    map_p: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {"TBS_TAB": self.tbs, "CB_TAB": self.cb, "SNR_MIN": self.snr_min,
                "SNR_MAX": np.array(self.snr_max), "SNR_HOLES": self.snr_holes,
                "MAP_LO": self.map_lo, "MAP_HI": self.map_hi, "MAP_P": self.map_p}


def snr_table(cr: dict[str, pd.DataFrame], pusch_prb: int) -> tuple[np.ndarray, int, np.ndarray]:
    """Per (MCS, nPRB): the lowest SNR measured on both decoders, the top SNR, and the holes (ADR-0020).

    CloudRIC keeps only decoded runs, so a measured (nPRB, SNR, MCS) triple is a decodable one.
    For every (MCS, nPRB) the SNRs measured on both decoders run from a minimum up to the
    dataset's top SNR; the few gaps inside that range are returned as `(mcs, nprb, snr)` rows.
    """
    common = set.intersection(*(set(zip(d.MCS.astype(int), d.nPRB.astype(int), d.SNR_dB.astype(int)))
                                for d in cr.values()))
    t = pd.DataFrame(sorted(common), columns=["mcs", "nprb", "snr"])
    snr_max = int(t.snr.max())
    g = t.groupby(["mcs", "nprb"]).snr.agg(["min", "max"])
    full = pd.MultiIndex.from_product([range(N_MCS), range(1, pusch_prb + 1)], names=["mcs", "nprb"])
    if not full.isin(g.index).all():
        raise ValueError("an (MCS, nPRB) has no SNR measured on both decoders")
    g = g.reindex(full)
    if (g["max"] != snr_max).any():
        raise ValueError("an (MCS, nPRB) is not measured up to the top SNR")
    have = set(zip(t.mcs, t.nprb, t.snr))
    holes = [(m, n, x) for (m, n), lo in g["min"].items() for x in range(lo, snr_max + 1) if (m, n, x) not in have]
    snr_min = g["min"].to_numpy().reshape(N_MCS, pusch_prb).astype(np.int8)
    return snr_min, snr_max, np.array(holes, dtype=np.int16).reshape(-1, 3)


def lte_to_nr_map(lte_bpp: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Option B (ADR-0018): scale LTE bits/PRB so LTE MCS 28 meets NR MCS 27, split between NR neighbours."""
    if lte_bpp.idxmax() != LTE_CEILING_MCS:
        raise ValueError("the LTE ceiling MCS 28 does not have the highest bits per PRB")
    scale = NR_EFF[-1] / lte_bpp[LTE_CEILING_MCS]
    lte_eff = (lte_bpp * scale).to_numpy()
    hi = np.clip(np.searchsorted(NR_EFF, lte_eff), 1, N_MCS - 1)
    lo = hi - 1
    p_hi = np.clip((lte_eff - NR_EFF[lo]) / (NR_EFF[hi] - NR_EFF[lo]), 0, 1)      # keeps the mean efficiency
    map_lo, map_hi, map_p = np.zeros(N_LTE_MCS, np.int8), np.zeros(N_LTE_MCS, np.int8), np.zeros(N_LTE_MCS)
    idx = lte_bpp.index.to_numpy()
    map_lo[idx], map_hi[idx], map_p[idx] = lo, hi, p_hi
    return map_lo, map_hi, map_p


def radio_config(cfg: TrafficConfig, prep_key: str) -> dict:
    """Everything stage R depends on (the key of its cache directory)."""
    d = cfg.to_dict()
    return {"cloudric_sha256": {f: cache.file_sha256(cfg.cloudric_dir / f"{f}_dataset.csv", cfg.cache_dir)
                                for f in CLOUDRIC_FILES},
            "prep": prep_key, "radio": {k: v for k, v in d["radio"].items() if k != "buffer_expiry_slots"},
            "version": GENERATOR_VERSION}                                  # the SNR draw (ADR-0020) is stage B


def build_radio(cfg: TrafficConfig, prep: Prep, cr: dict[str, pd.DataFrame] | None = None, key: str = "") -> Radio:
    """Compute stage R in memory."""
    cr = read_cloudric(cfg.cloudric_dir) if cr is None else cr
    rc = cfg.radio
    tbs = tbs_table(rc.pusch_prb, rc.re_per_prb, rc.layers)
    snr_min, snr_max, holes = snr_table(cr, rc.pusch_prb)
    return Radio(key, tbs, cb_table(tbs), snr_min, snr_max, holes, *lte_to_nr_map(prep.lte_bits_per_prb))


def save_radio(radio: Radio, d: Path, config: dict) -> None:
    np.savez(d / "tables.npz", **radio.arrays())
    pd.DataFrame(radio.snr_min, index=pd.Index(range(N_MCS), name="mcs"),     # readable copy of the ADR-0020 table
                 columns=range(1, radio.snr_min.shape[1] + 1)).to_csv(d / "snr_min.csv")
    pd.DataFrame(radio.snr_holes, columns=["mcs", "nprb", "snr_db"]).to_csv(d / "snr_holes.csv", index=False)
    (d / cache.MANIFEST).write_text(json.dumps({"key": radio.key, "config": config}, indent=1))


def load_radio(d: Path) -> Radio:
    key = json.loads((d / cache.MANIFEST).read_text())["key"]
    with np.load(d / "tables.npz") as z:
        a = {k: z[k] for k in z.files}
    return Radio(key, a["TBS_TAB"], a["CB_TAB"], a["SNR_MIN"], int(a["SNR_MAX"]), a["SNR_HOLES"], a["MAP_LO"], a["MAP_HI"], a["MAP_P"])


def radio_dir(cfg: TrafficConfig, prep_key: str) -> Path:
    return cfg.cache_dir / "radio_tables" / cache.key(radio_config(cfg, prep_key))


def ensure_radio(cfg: TrafficConfig, prep: Prep | None = None) -> Radio:
    """Stage R from the cache, building it (and stage A) first if missing. Memoized per process."""
    prep = ensure_prep(cfg) if prep is None else prep
    return _ensure_radio(cfg, radio_dir(cfg, prep.key), prep)


@lru_cache(maxsize=4)
def _ensure_radio(cfg: TrafficConfig, d: Path, prep: Prep) -> Radio:
    if not cache.is_complete(d):
        config = radio_config(cfg, prep.key)
        radio = build_radio(cfg, prep, key=d.name)
        cache.write_once(d, lambda tmp: save_radio(radio, tmp, config))
    return load_radio(d)
