"""Stages B1-B3: requests, round-robin radio scheduler with buffer expiry, reception.

docs/TRAFFIC.md §6-§8; ADR-0016, ADR-0018, ADR-0019, ADR-0020. Slots are integer indices;
times become ms only in the output.
"""
from __future__ import annotations

import bisect
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from acmad.nr import MCS_T2

from .config import TrafficConfig
from .radio import Radio
from .stream import JOB_COLUMNS, TASK_COLUMNS

SEED_PURPOSES = ("scenario", "snr", "slot", "mcs")   # SeedSequence children, in this order; append only


def rngs(seed: int) -> dict[str, np.random.Generator]:
    """One independent generator per purpose, from `SeedSequence(seed).spawn` (TRAFFIC.md §11.3)."""
    children = np.random.SeedSequence(seed).spawn(len(SEED_PURPOSES))
    return dict(zip(SEED_PURPOSES, map(np.random.default_rng, children)))


# ---- B1. requests ------------------------------------------------------------------------

def requests(rows: dict[str, np.ndarray], radio: Radio, slots_per_tti: int, rng_mcs: np.random.Generator,
             rng_slot: np.random.Generator) -> dict[str, np.ndarray]:
    """One request per row of a DU: `(slot, ue, bits, mcs, src_block)`, sorted by slot (stable).

    NR MCS by option B (stream `mcs`) and the 1st or 2nd slot of the TTI 50 / 50 (stream
    `slot`). Row times are whole TTIs. The SNR is drawn later, per grant (ADR-0020).
    """
    n = len(rows["TBS"])
    lte = rows["MCS"]
    mcs = np.where(rng_mcs.random(n) < radio.map_p[lte], radio.map_hi[lte], radio.map_lo[lte]).astype(np.int8)
    slot = rows["t_ms"].astype(np.int64) * slots_per_tti + rng_slot.integers(0, slots_per_tti, n)
    order = np.argsort(slot, kind="stable")
    return {"slot": slot[order], "ue": rows["RNTI"].astype(np.int64)[order], "bits": rows["bits"][order],
            "mcs": mcs[order], "src_block": rows["src_block"][order]}


# ---- B2. scheduler -----------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Grants:
    """Grants of one DU (internal, ADR-0016). Indices `oldest` / `latest` point into the requests."""

    slot: np.ndarray        # grant slot t0 (index)
    nprb: np.ndarray
    mcs: np.ndarray
    tbs: np.ndarray
    served: np.ndarray      # bits taken out of the buffer (<= tbs)
    oldest: np.ndarray      # request of the oldest bits still buffered at the grant
    latest: np.ndarray      # latest request of the user (gives the MCS)
    ue: np.ndarray

    def __len__(self) -> int:
        return len(self.slot)


def schedule(req: dict[str, np.ndarray], n_slots: int, tbs_tab: np.ndarray, pusch_prb: int,
             expiry_slots: int) -> tuple[Grants, np.ndarray, dict]:
    """Round robin over active users with one buffer each, per slot (TRAFFIC.md §7).

    Per slot: 0. bits requested `expiry_slots` slots ago or earlier are dropped (ADR-0019);
    1. the slot's requests join their users' buffers (new users at the end of the list);
    2. each user in turn gets the fewest PRBs whose TBS carries its buffer, or what is left;
    3. if the slot filled up, the next one starts with the user whose turn was cut.
    Returns the grants, the PRBs used per slot and the bit counts.
    """
    slot_of = req["slot"].tolist()
    ue_of = req["ue"].tolist()
    bits_of = req["bits"].tolist()
    mcs_of = req["mcs"].astype(int).tolist()
    tbs_rows = [row.tolist() for row in tbs_tab]
    starts = np.searchsorted(req["slot"], np.arange(n_slots + 1)).tolist()
    buf: dict[int, deque] = {}      # user -> deque of [request, bits left]
    tot: dict[int, int] = {}        # user -> bits buffered
    last: dict[int, int] = {}       # user -> latest request
    active: list[int] = []          # round-robin list
    rows: list[tuple[int, ...]] = []
    used = np.zeros(n_slots, dtype=np.int64)
    expired = 0
    for t in range(n_slots):
        for u in active:                                          # 0. expiry
            q = buf[u]
            while q and slot_of[q[0][0]] <= t - expiry_slots:
                expired += q[0][1]
                tot[u] -= q[0][1]
                q.popleft()
        if any(not tot[u] for u in active):
            for u in [u for u in active if not tot[u]]:
                del buf[u], tot[u]
            active = [u for u in active if u in buf]
        for i in range(starts[t], starts[t + 1]):                # 1. arrivals
            u = ue_of[i]
            if u not in buf:
                buf[u] = deque()
                tot[u] = 0
                active.append(u)
            buf[u].append([i, bits_of[i]])
            tot[u] += bits_of[i]
            last[u] = i
        free, cut = pusch_prb, None                               # 2. round robin
        for j, u in enumerate(active):
            if free == 0:
                cut = j
                break
            q, m, oldest = buf[u], mcs_of[last[u]], buf[u][0][0]
            row = tbs_rows[m]
            need = bisect.bisect_left(row, tot[u]) + 1
            nprb = min(need, free)
            tbs = row[nprb - 1]
            served = min(tbs, tot[u])
            tot[u] -= served
            left = served
            while left:
                take = min(q[0][1], left)
                q[0][1] -= take
                left -= take
                if not q[0][1]:
                    q.popleft()
            rows.append((t, nprb, m, tbs, served, oldest, last[u], u))
            free -= nprb
            if nprb < need:                                       # slot full during u's turn
                cut = j if tot[u] else j + 1
                break
        used[t] = pusch_prb - free
        if cut:                                                   # 3. rotate
            active = active[cut:] + active[:cut]
        if any(not tot[u] for u in active):
            for u in [u for u in active if not tot[u]]:
                del buf[u], tot[u]
            active = [u for u in active if u in buf]
    leftover = sum(tot.values())

    cols = [np.array(c, dtype=np.int64) for c in zip(*rows)] if rows else [np.zeros(0, np.int64)] * 8
    grants = Grants(*cols)
    bits_requested = int(sum(bits_of))
    stats = {"requests": len(slot_of), "granted": len(grants), "bits_requested": bits_requested,
             "bits_served": int(grants.served.sum()), "bits_expired": int(expired), "bits_left_over": int(leftover),
             "prb_util_mean": float(used.mean() / pusch_prb) if n_slots else 0.0,
             "prb_util_p99": float(np.percentile(used, 99) / pusch_prb) if n_slots else 0.0,
             "wait_slots_p99": float(np.percentile(grants.slot - req["slot"][grants.oldest], 99)) if len(grants) else 0.0}
    if stats["bits_served"] + stats["bits_expired"] + stats["bits_left_over"] != bits_requested:
        raise AssertionError("scheduler lost bits")
    return grants, used, stats


# ---- B3. reception -----------------------------------------------------------------------

def draw_snr(mcs: np.ndarray, nprb: np.ndarray, radio: Radio, decay_db: float,
             rng: np.random.Generator) -> np.ndarray:
    """Task SNR per grant (ADR-0020): `snr_min(m, n) + k`, k >= 0 with P(k) ∝ exp(-k / decay_db).

    k runs over the SNRs measured on both CloudRIC decoders for (m, n): up to `snr_max`,
    minus the holes, so every (nPRB, SNR, MCS) is a measured triple. `decay_db = 0` gives
    the minimum always. Inverse CDF of the truncated geometric; draws that hit a hole are
    redrawn (which renormalises over the measured SNRs).
    """
    lo = radio.snr_min[mcs, nprb - 1].astype(np.int64)
    if decay_db == 0:
        return lo.astype(np.int8)
    q = np.exp(-1.0 / decay_db)
    k_max = radio.snr_max - lo
    holes = {tuple(h) for h in radio.snr_holes.tolist()}
    snr = np.empty_like(lo)
    todo = np.arange(len(lo))
    while len(todo):
        u = rng.random(len(todo))
        k = np.ceil(np.log1p(-u * (1 - q ** (k_max[todo] + 1))) / np.log(q)) - 1
        snr[todo] = lo[todo] + np.clip(k, 0, k_max[todo]).astype(np.int64)
        todo = todo[[(int(mcs[i]), int(nprb[i]), int(snr[i])) in holes for i in todo]] if holes else todo[:0]
    return snr.astype(np.int8)


def reception(grants: list[Grants], radio: Radio, cfg: TrafficConfig,
              rng_snr: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every grant -> one job (arrival at the PUSCH end + fronthaul + jitter) and two tasks (ADR-0020, ADR-0016).

    The task SNR is drawn here, per grant, from its (MCS, nPRB) (ADR-0020; stream `snr`).

    Jobs are sorted by `t_arrival_ms` (stable over DU, then grant order) and numbered in
    that order; task `2j` is the cpu task of job j, `2j + 1` its ldpc task.
    """
    r, tm = cfg.radio, cfg.timing
    du = np.concatenate([np.full(len(g), d, dtype=np.int16) for d, g in enumerate(grants)])
    slot = np.concatenate([g.slot for g in grants])
    t_pusch_end = slot * r.slot_ms + tm.k2_slots * r.slot_ms + r.slot_ms
    t_arrival = t_pusch_end + tm.fronthaul_ms + tm.jitter_ms
    nprb = np.concatenate([g.nprb for g in grants]).astype(np.int16)
    mcs = np.concatenate([g.mcs for g in grants]).astype(np.int8)
    cb = radio.cb[mcs, nprb - 1]                                             # (base graph, n_cb, cb_len), ADR-0019
    qm_r = np.array(MCS_T2)[mcs]
    feats = {"nprb": nprb, "mcs": mcs,
             "snr_db": draw_snr(mcs.astype(np.int64), nprb.astype(np.int64), radio, cfg.channel.snr_decay_db, rng_snr),
             "tbs": np.concatenate([g.tbs for g in grants]).astype(np.int32),
             "base_graph": cb[:, 0].astype(np.int8), "n_cb": cb[:, 1].astype(np.int16),
             "cb_len": cb[:, 2].astype(np.int32), "total_bits": (cb[:, 1] * cb[:, 2]).astype(np.int32),
             "qm": qm_r[:, 0].astype(np.int8), "code_rate": (qm_r[:, 1] / 1024).astype(np.float32)}
    order = np.argsort(t_arrival, kind="stable")
    nj = len(order)
    ids = np.arange(nj, dtype=np.int64)
    jobs = pd.DataFrame({"job_id": ids, "du": du[order], "t_arrival_ms": t_arrival[order]})
    depends_on = pd.array(np.repeat(ids * 2, 2), dtype="Int64")
    depends_on[::2] = pd.NA                                                  # the cpu task depends on nothing
    tasks = pd.DataFrame({"task_id": np.arange(2 * nj, dtype=np.int64), "job_id": np.repeat(ids, 2),
                          "kind": pd.Categorical(np.tile(["cpu_processing", "ldpc_decoding"], nj),
                                                 categories=["cpu_processing", "ldpc_decoding"]),
                          "allowed": pd.Categorical(np.tile(["cpu", "any"], nj), categories=["any", "cpu"]),
                          "depends_on": depends_on, **{k: np.repeat(v[order], 2) for k, v in feats.items()}})
    assert dict(jobs.dtypes.astype(str)) == JOB_COLUMNS and dict(tasks.dtypes.astype(str)) == TASK_COLUMNS
    return jobs, tasks
