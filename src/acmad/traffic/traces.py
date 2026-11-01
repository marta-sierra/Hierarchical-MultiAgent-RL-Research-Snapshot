"""Stage A: read the trace, preprocess it (A1-A5b), cut blocks / super-blocks, draw the split,
label the load classes.

docs/TRAFFIC.md §4 and §11.2; ADR-0017, ADR-0018, ADR-0007. The only module that reads the trace CSV.
Cached once in `data/cache/traffic_prep/<hash>/` and memory-mapped by every worker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from acmad import cache

from .config import GENERATOR_VERSION, LOAD_CLASSES, TTI_MS, TrafficConfig

ROW_COLUMNS: dict[str, type] = {"t_ms": np.int32, "RNTI": np.uint16, "MCS": np.int8, "PRB": np.int16, "TBS": np.int32}
RETX_MIN_LTE_MCS = 29           # A5: LTE MCS 29-31 are HARQ retransmissions


@dataclass(frozen=True, eq=False)
class Prep:
    """Stage A output.

    `rows[c][block_ptr[b]:block_ptr[b+1]]` are the rows of block b; `t_ms` is whole ms from
    the block start. `blocks` / `superblocks` hold each one's load (Mb/s), split and load class.
    `lte_bits_per_prb`: median TBS / PRB per LTE MCS (single codeword, >= 20 PRBs), the
    input of the option-B LTE -> NR MCS map (ADR-0018).
    """

    key: str
    rows: dict[str, np.ndarray]
    block_ptr: np.ndarray
    blocks: pd.DataFrame
    superblocks: pd.DataFrame
    lte_bits_per_prb: pd.Series
    stats: dict

    def block_rows(self, b: int) -> dict[str, np.ndarray]:
        """Rows of block b (O(1) slices of the columns)."""
        sl = slice(int(self.block_ptr[b]), int(self.block_ptr[b + 1]))
        return {c: np.asarray(v[sl]) for c, v in self.rows.items()}


def prep_config(cfg: TrafficConfig) -> dict:
    """Everything stage A depends on (the key of its cache directory)."""
    return {"trace_sha256": cache.file_sha256(cfg.trace_path, cfg.cache_dir),
            "trace": {k: v for k, v in cfg.to_dict()["trace"].items() if k != "path"},
            "split": cfg.to_dict()["split"], "version": GENERATOR_VERSION}


def prep_key(cfg: TrafficConfig) -> str:
    return cache.key(prep_config(cfg))


def stratified_split(loads: np.ndarray, stratum_size: int, per_stratum: dict[str, int], seed: int) -> np.ndarray:
    """A7: sort by load, strata of `stratum_size`, permute the split pattern in each; an incomplete last stratum -> train."""
    rng = np.random.default_rng(seed)
    order = np.argsort(-np.asarray(loads), kind="stable")
    pattern = [s for s, n in per_stratum.items() for _ in range(n)]
    out = np.empty(len(loads), dtype=object)
    for i in range(0, len(order), stratum_size):
        members = order[i: i + stratum_size]
        out[members] = "train" if len(members) < stratum_size else rng.permutation(pattern)
    return out


def load_classes(loads: np.ndarray, stratum_size: int) -> np.ndarray:
    """A8 (ADR-0007): whole strata of A7 (same load order) -> high / medium / low.

    The strata are shared out as evenly as possible, an extra one going to both extremes
    first (11 strata -> 4 / 3 / 4), so every class keeps the split pattern of its strata.
    """
    order = np.argsort(-np.asarray(loads), kind="stable")
    n_strata = -(-len(order) // stratum_size)
    base, rem = divmod(n_strata, 3)
    n_high, n_medium = base + (rem == 2), base + (rem == 1)
    stratum = np.empty(len(order), dtype=np.int64)
    stratum[order] = np.arange(len(order)) // stratum_size                      # 0 = highest load
    low, medium, high = LOAD_CLASSES
    return np.where(stratum < n_high, high, np.where(stratum < n_high + n_medium, medium, low)).astype(object)


def preprocess(raw: pd.DataFrame, cfg: TrafficConfig) -> tuple[pd.DataFrame, int, dict]:
    """A1-A5b. Returns the rows `(tti, RNTI, MCS, PRB, TBS, n_cw, t_ms)`, the TTI count and stats."""
    n_raw = len(raw)
    raw = raw.drop_duplicates()                                                  # A1
    tti = raw["SFN"] * 10 + raw["Subframe"]                                      # A2
    raw = raw.assign(tti=tti - tti.min())
    u = raw[~raw["RNTI"].isin(cfg.trace.drop_rntis)]                             # A3
    g = (u.groupby(["tti", "RNTI"], sort=False)                                  # A4 (ADR-0018: MCS = max)
           .agg(MCS=("MCS", "max"), PRB=("PRB", "max"), TBS=("TBS", "sum"), n_cw=("TBS", "size"))
           .reset_index().sort_values(["tti", "RNTI"], kind="stable").reset_index(drop=True))
    retx = g["MCS"] >= RETX_MIN_LTE_MCS                                          # A5
    stats = {"rows_raw": n_raw, "duplicates_dropped": n_raw - len(raw),
             "broadcast_fraction": float(1 - len(u) / len(raw)),
             "two_codeword_pairs_merged": int((g.n_cw == 2).sum()),
             "retransmission_fraction": float(retx.mean())}
    g = g[~retx]
    zero = g["TBS"] == 0                                                         # A5b (ADR-0018): no bits to carry
    stats["zero_tbs_fraction"] = float(zero.mean())
    g = g[~zero].copy()
    g["MCS"] = g["MCS"].astype("int16")
    g["t_ms"] = g["tti"] * TTI_MS
    return g, int(raw["tti"].max()) + 1, stats


def build_prep(cfg: TrafficConfig, raw: pd.DataFrame | None = None) -> Prep:
    """Compute stage A in memory (reads the trace CSV unless `raw` is given)."""
    if raw is None:
        raw = pd.read_csv(cfg.trace_path)
        pkey = prep_key(cfg)
    else:
        pkey = cache.key({"raw": "in-memory", "cfg": cfg.to_dict()["split"]})
    g, n_tti, stats = preprocess(raw, cfg)
    sp = cfg.split
    block_ms = sp.block_ms
    bits = np.bincount(g["tti"], weights=g["TBS"], minlength=n_tti)
    w = int(sp.block_s * 1000)
    blk = bits[: n_tti // w * w].reshape(-1, w).sum(1) / sp.block_s / 1e6        # Mb/s of each block
    bps = sp.blocks_per_superblock
    n_blk = n_tti // (sp.superblock_s * 1000) * bps                              # A6: trailing partial super-block dropped
    blocks = pd.DataFrame({"block": np.arange(n_blk), "superblock": np.arange(n_blk) // bps,
                           "start_ms": np.arange(n_blk) * block_ms, "mbps": blk[:n_blk]})
    superblocks = blocks.groupby("superblock").agg(start_ms=("start_ms", "min"), mbps=("mbps", "mean")).reset_index()
    superblocks["split"] = stratified_split(superblocks["mbps"], sp.stratum_size, dict(sp.per_stratum), sp.split_seed)
    superblocks["load_class"] = load_classes(superblocks["mbps"], sp.stratum_size)
    blocks = blocks.merge(superblocks[["superblock", "split", "load_class"]], on="superblock")
    g = g[g["t_ms"] < n_blk * block_ms].copy()
    g["block"] = (g["t_ms"] // block_ms).astype(np.int64)                        # bursts across a block edge are cut
    assert (np.diff(g.block.to_numpy()) >= 0).all()
    rows = {"t_ms": (g.t_ms - g.block * block_ms).to_numpy(np.int32), "RNTI": g.RNTI.to_numpy(np.uint16),
            "MCS": g.MCS.to_numpy(np.int8), "PRB": g.PRB.to_numpy(np.int16), "TBS": g.TBS.to_numpy(np.int32)}
    block_ptr = np.searchsorted(g.block.to_numpy(), np.arange(n_blk + 1)).astype(np.int64)
    single = g[(g.n_cw == 1) & (g.PRB >= 20)]
    lte_bpp = (single.TBS / single.PRB).groupby(single.MCS).median()
    if set(lte_bpp.index) != set(g.MCS.unique()):
        raise ValueError("an LTE MCS of the rows has no single-codeword row with >= 20 PRBs (option-B map)")
    stats.update(rows=int(block_ptr[-1]), n_tti=n_tti, blocks=int(n_blk), superblocks=len(superblocks))
    return Prep(pkey, rows, block_ptr, blocks, superblocks, lte_bpp, stats)


def save_prep(prep: Prep, d: Path, config: dict) -> None:
    """Write stage A into directory `d` (manifest last)."""
    for c, v in prep.rows.items():
        np.save(d / f"rows_{c}.npy", v)
    np.save(d / "block_ptr.npy", prep.block_ptr)
    prep.blocks.to_parquet(d / "blocks.parquet")
    prep.superblocks.to_parquet(d / "superblocks.parquet")
    (d / "lte_bits_per_prb.json").write_text(json.dumps({int(k): float(v) for k, v in prep.lte_bits_per_prb.items()}))
    (d / cache.MANIFEST).write_text(json.dumps({"key": prep.key, "config": config, "stats": prep.stats}, indent=1))


def load_prep(d: Path) -> Prep:
    """Open stage A read-only; the row columns are memory-mapped."""
    manifest = json.loads((d / cache.MANIFEST).read_text())
    rows = {c: np.load(d / f"rows_{c}.npy", mmap_mode="r") for c in ROW_COLUMNS}
    bpp = json.loads((d / "lte_bits_per_prb.json").read_text())
    lte_bpp = pd.Series({int(k): float(v) for k, v in sorted(bpp.items(), key=lambda kv: int(kv[0]))}, dtype=float)
    lte_bpp.index.name = "MCS"
    return Prep(manifest["key"], rows, np.load(d / "block_ptr.npy"), pd.read_parquet(d / "blocks.parquet"),
                pd.read_parquet(d / "superblocks.parquet"), lte_bpp, manifest["stats"])


def prep_dir(cfg: TrafficConfig) -> Path:
    return cfg.cache_dir / "traffic_prep" / prep_key(cfg)


def ensure_prep(cfg: TrafficConfig) -> Prep:
    """Stage A from the cache, building it first if missing. Memoized per process."""
    return _ensure_prep(cfg, prep_dir(cfg))


@lru_cache(maxsize=4)
def _ensure_prep(cfg: TrafficConfig, d: Path) -> Prep:
    if not cache.is_complete(d):
        config = prep_config(cfg)
        prep = build_prep(cfg)
        cache.write_once(d, lambda tmp: save_prep(prep, tmp, config))
    return load_prep(d)
