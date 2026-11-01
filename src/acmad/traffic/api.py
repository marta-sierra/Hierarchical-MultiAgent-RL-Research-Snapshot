"""`generate(cfg, split, n_dus, s, seed, load_class=None) -> TaskStream` (docs/TRAFFIC.md §11.1)."""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from acmad import cache

from .config import GENERATOR_VERSION, LOAD_CLASSES, MCS_TABLE, SPLITS, TrafficConfig
from .radio import Radio, ensure_radio
from .scaling import build_episode
from .stream import TaskStream
from .traces import Prep, ensure_prep
from .workload import SEED_PURPOSES, reception, requests, rngs, schedule

LOAD_TOTALS = ("requests", "granted", "bits_requested", "bits_served", "bits_expired", "bits_left_over")


def generate(cfg: TrafficConfig, split: str, n_dus: int, s: float, seed: int,
             load_class: str | None = None) -> TaskStream:
    """One episode of traffic: `n_dus` cells from one super-block of `split`, bits x `s`.

    `load_class` (`low` / `medium` / `high`, ADR-0007) restricts the super-block to that
    class; None draws from every class of the split (same streams as without the option).
    Pure: the same `(cfg, split, n_dus, s, seed, load_class)` gives the same stream. Stages A and R
    come from the cache (built on first use; the trace is read only then).
    """
    prep = ensure_prep(cfg)
    return generate_from(prep, ensure_radio(cfg, prep), cfg, split, n_dus, s, seed, load_class)


def generate_from(prep: Prep, radio: Radio, cfg: TrafficConfig, split: str, n_dus: int, s: float,
                  seed: int, load_class: str | None = None) -> TaskStream:
    """`generate` on given stage-A / stage-R objects (in memory or loaded from the cache)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if load_class is not None and load_class not in LOAD_CLASSES:
        raise ValueError(f"load_class must be None or one of {LOAD_CLASSES}, got {load_class!r}")
    if n_dus < 1 or not s > 0:
        raise ValueError("n_dus must be >= 1 and s > 0")
    r = rngs(seed)
    rows_per_du, scen = build_episode(prep, split, n_dus, s, r["scenario"], load_class)
    rc = cfg.radio
    T_ms = float(cfg.split.block_ms)
    n_slots = int(round(T_ms / rc.slot_ms))
    grants, used, per_du = [], [], []
    for rows in rows_per_du:                                              # one scheduler per DU
        req = requests(rows, radio, rc.slots_per_tti, r["mcs"], r["slot"])
        gr, u, st = schedule(req, n_slots, radio.tbs, rc.pusch_prb, rc.buffer_expiry_slots)
        grants.append(gr); used.append(u); per_du.append(st)
    jobs, tasks = reception(grants, radio, cfg, r["snr"])

    T_s = T_ms / 1e3
    for st, g in zip(per_du, grants):
        st["wait_ms_p99"] = st.pop("wait_slots_p99") * rc.slot_ms
        st["requested_mbps"] = st["bits_requested"] / T_s / 1e6
        st["granted_tbs_mbps"] = int(g.tbs.sum()) / T_s / 1e6
        st["jobs_per_s"] = len(g) / T_s
    tot = {c: sum(st[c] for st in per_du) for c in LOAD_TOTALS}
    all_used = np.concatenate(used)
    meta = {
        **scen, "seed": seed,
        "seed_layout": {"entropy": seed, "spawn": list(SEED_PURPOSES)},
        "generator_version": GENERATOR_VERSION, "git_hash": git_hash(),
        "config": cfg.to_dict(), "config_hash": cfg.hash(),
        "prep_key": prep.key, "radio_key": radio.key,
        "T_ms": T_ms, "slot_ms": rc.slot_ms, "carrier_prb": rc.carrier_prb, "slot_prb": rc.pusch_prb,
        "re_per_prb": rc.re_per_prb, "layers": rc.layers, "mcs_table": MCS_TABLE,
        "lte_to_nr_mcs": rc.lte_to_nr_mcs, "buffer_expiry_slots": rc.buffer_expiry_slots,
        "snr_decay_db": cfg.channel.snr_decay_db, "snr_max": radio.snr_max,
        "k2_slots": cfg.timing.k2_slots, "fronthaul_ms": cfg.timing.fronthaul_ms, "jitter_ms": cfg.timing.jitter_ms,
        "pending_placeholders": {},
        "requested_mbps": tot["bits_requested"] / T_s / 1e6,
        "granted_tbs_mbps": int(sum(int(g.tbs.sum()) for g in grants)) / T_s / 1e6,
        "jobs_per_s": len(jobs) / T_s,
        **tot,
        "prb_util_mean": float(np.mean([st["prb_util_mean"] for st in per_du])),
        "prb_util_p99": float(np.percentile(all_used, 99) / rc.pusch_prb),
        "wait_ms_p99": max(st["wait_ms_p99"] for st in per_du),
        "per_du": per_du,
    }
    return TaskStream(jobs, tasks, meta)


def stream_key(cfg: TrafficConfig, split: str, n_dus: int, s: float, seed: int,
               load_class: str | None = None) -> str:
    """Key of a fixed stream in `data/cache/traffic/<key>/`."""
    prep = ensure_prep(cfg)
    radio = ensure_radio(cfg, prep)
    t = cfg.to_dict()["timing"]
    return cache.key({"prep": prep.key, "radio": radio.key, "timing": t,
                      "expiry": cfg.radio.buffer_expiry_slots, "snr_decay_db": float(cfg.channel.snr_decay_db),
                      "split": split, "load_class": load_class, "n_dus": int(n_dus),
                      "s": float(s), "seed": int(seed), "version": GENERATOR_VERSION})


def stream_dir(cfg: TrafficConfig, split: str, n_dus: int, s: float, seed: int,
               load_class: str | None = None) -> Path:
    return cfg.cache_dir / "traffic" / stream_key(cfg, split, n_dus, s, seed, load_class)


def load_or_generate(cfg: TrafficConfig, split: str, n_dus: int, s: float, seed: int,
                     load_class: str | None = None) -> TaskStream:
    """A fixed stream: read from `data/cache/traffic/<key>/`, generated and saved first if missing."""
    d = stream_dir(cfg, split, n_dus, s, seed, load_class)
    if not cache.is_complete(d):
        st = generate(cfg, split, n_dus, s, seed, load_class)

        def write(tmp: Path) -> None:
            st.save(tmp)
            (tmp / cache.MANIFEST).write_text((tmp / "meta.json").read_text())

        cache.write_once(d, write)
    return TaskStream.load(d)


@lru_cache(maxsize=1)
def git_hash() -> str | None:
    """Commit of the working tree (with `-dirty`), or None outside git."""
    root = Path(__file__).resolve().parents[3]
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        if h.returncode:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return h.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None
