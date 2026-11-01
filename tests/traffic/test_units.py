"""Fast unit tests on small synthetic inputs (no trace needed)."""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from acmad import cache
from acmad.traffic.config import TrafficConfig, load_config
from acmad.nr import code_blocks, tbs_bits, tbs_table
from acmad.traffic.radio import Radio
from acmad.traffic.scaling import assign_blocks
from acmad.traffic.stream import TaskStream
from acmad.traffic.traces import build_prep, load_classes, stratified_split
from acmad.traffic.workload import Grants, draw_snr, reception, requests, rngs, schedule

CFG = TrafficConfig()
TBS = tbs_table(CFG.radio.pusch_prb, CFG.radio.re_per_prb, CFG.radio.layers)


def req(slot, ue, bits, mcs=21):
    n = len(slot)
    return {"slot": np.array(slot, np.int64), "ue": np.array(ue, np.int64), "bits": np.array(bits, np.int64),
            "mcs": np.full(n, mcs, np.int8), "src_block": np.zeros(n, np.int64)}


# ---- config ------------------------------------------------------------------------------

def test_default_yaml_equals_dataclass_defaults():
    assert load_config() == TrafficConfig()
    assert load_config().hash() == TrafficConfig().hash()


@pytest.mark.parametrize("section,key,value", [("radio", "lte_to_nr_mcs", "A"), ("trace", "codeword_mcs", "first"),
                                               ("radio", "buffer_expiry_slots", 0), ("radio", "mcs_table", "T1")])
def test_config_refuses_undecided_choices(section, key, value):
    d = TrafficConfig().to_dict()
    d[section][key] = value
    with pytest.raises(ValueError):
        TrafficConfig.from_dict(d)


def test_config_refuses_unknown_keys():
    with pytest.raises(ValueError):
        TrafficConfig.from_dict({"radio": {"nope": 1}})


# ---- TBS / code blocks (spot values; the full check against CloudRIC is in test_generator) ----

def test_tbs_monotone_in_prbs():
    assert (np.diff(TBS, axis=1) >= 0).all()
    assert TBS.shape == (28, 250)


def test_tbs_spot_values():
    assert tbs_bits(0, 1) == 32           # CloudRIC CPU_dataset first row: nPRB 1, MCS 0 -> TBS 32
    assert code_blocks(32, 0) == (2, 1, 80)


# ---- seeds -------------------------------------------------------------------------------

def test_seed_children_order_is_fixed():
    r = rngs(5)
    assert list(r) == ["scenario", "snr", "slot", "mcs"]
    kids = np.random.SeedSequence(5).spawn(4)
    for (name, g), k in zip(r.items(), kids):
        assert g.random() == np.random.default_rng(k).random(), name


# ---- split and blocks --------------------------------------------------------------------

def test_stratified_split_one_per_stratum():
    loads = np.arange(12, dtype=float)
    out = stratified_split(loads, 5, {"train": 3, "val": 1, "test": 1}, seed=0)
    order = np.argsort(-loads, kind="stable")
    for i in (0, 5):
        assert sorted(out[order[i:i + 5]]) == ["test", "train", "train", "train", "val"]
    assert list(out[order[10:]]) == ["train", "train"]      # incomplete last stratum


@pytest.mark.parametrize("n, sizes", [(55, (20, 15, 20)), (45, (15, 15, 15)), (52, (20, 15, 17)),
                                      (20, (5, 10, 5)), (10, (5, 0, 5)), (2, (0, 2, 0))])
def test_load_classes_follow_whole_strata(n, sizes):
    """High / medium / low = whole strata of A7, extras to the extremes; the incomplete one is low."""
    loads = np.random.default_rng(1).permutation(n).astype(float)
    out = load_classes(loads, 5)
    order = np.argsort(-loads, kind="stable")
    n_high, n_medium, _ = sizes
    assert list(out[order]) == ["high"] * n_high + ["medium"] * n_medium + ["low"] * (n - n_high - n_medium)


def test_assign_blocks_least_used():
    cand = np.array([10, 11, 12])
    per_du = assign_blocks(cand, 7, np.random.default_rng(0))
    counts = pd.Series(per_du).value_counts()
    assert set(per_du[:3]) == {10, 11, 12} and counts.max() - counts.min() <= 1


def synthetic_raw(n_s=240, seed=0):
    """A tiny trace in the CSV's format: 2 users per TTI, a broadcast row, a duplicate and a retransmission."""
    rng = np.random.default_rng(seed)
    n = n_s * 1000
    tti = np.repeat(np.arange(n), 2)
    rnti = np.tile([100, 200], n)
    mcs = rng.integers(0, 29, 2 * n)
    mcs[:58] = np.repeat(np.arange(29), 2)
    df = pd.DataFrame({"Entry": np.arange(2 * n), "Timestamp": 0.0, "SFN": tti // 10, "Subframe": tti % 10,
                       "RNTI": rnti, "Direction": 1, "MCS": mcs, "PRB": 25, "TBS": 25 * (mcs + 1) * 10})
    extra = pd.DataFrame({"Entry": [-1, -2, -3], "Timestamp": 0.0, "SFN": [0, 0, 0], "Subframe": [0, 1, 2],
                          "RNTI": [65535, 300, 400], "Direction": 1, "MCS": [0, 30, 5], "PRB": [4, 4, 4],
                          "TBS": [100, 100, 100]})
    return pd.concat([df.iloc[:10], df, extra], ignore_index=True)   # first 10 rows duplicated (A1)


def test_build_prep_synthetic():
    prep = build_prep(CFG, raw=synthetic_raw())
    assert prep.stats["duplicates_dropped"] == 10
    assert prep.stats["blocks"] == 12 and prep.stats["superblocks"] == 2
    rnti = np.asarray(prep.rows["RNTI"])
    assert 65535 not in rnti and 300 not in rnti            # broadcast (A3) and retransmission (A5) dropped
    assert 400 in rnti
    t = np.asarray(prep.rows["t_ms"])
    for b in range(prep.stats["blocks"]):
        blk = t[prep.block_ptr[b]:prep.block_ptr[b + 1]]
        assert blk.min() >= 0 and blk.max() < CFG.split.block_ms and (np.diff(blk) >= 0).all()


# ---- scheduler ---------------------------------------------------------------------------

def test_worked_example():
    """Notebook §5: A 100 kbit, B 150 kbit, C 80 kbit in slot 0; B 30 kbit more in slot 1; MCS 21."""
    r = req([0, 0, 0, 1], [1, 2, 3, 2], [100_000, 150_000, 80_000, 30_000])
    g, used, st = schedule(r, 4, TBS, 250, 4)
    assert (g.nprb <= 250).all() and (g.nprb >= 1).all()
    assert (used <= 250).all()
    assert st["bits_served"] == st["bits_requested"] == 360_000 and st["bits_expired"] == 0
    assert g.ue[0] == 1 and g.slot[0] == 0
    # user A's whole buffer fits: fewest PRBs whose TBS carries 100 kbit
    need = int(np.searchsorted(TBS[21], 100_000)) + 1
    assert g.nprb[0] == need and g.tbs[0] >= 100_000 > TBS[21, need - 2]
    # slot 0 fills up during B's turn; slot 1 starts with B
    s0 = g.slot == 0
    assert g.nprb[s0].sum() == 250 and list(g.ue[s0]) == [1, 2]
    assert g.ue[g.slot == 1][0] == 2


def test_expiry_after_four_slots():
    """One user at MCS 0 asks far more than 4 slots carry: what is not granted by slot 3 expires."""
    r = req([0], [1], [10_000_000], mcs=0)
    g, used, st = schedule(r, 10, TBS, 250, 4)
    assert list(g.slot) == [0, 1, 2, 3] and (g.nprb == 250).all()
    assert st["bits_expired"] == 10_000_000 - 4 * TBS[0, 249] and st["bits_left_over"] == 0
    assert (used[4:] == 0).all()


def test_left_over_counted_at_end():
    r = req([8], [1], [10_000_000], mcs=0)
    g, _, st = schedule(r, 10, TBS, 250, 4)
    assert len(g) == 2 and st["bits_left_over"] == 10_000_000 - 2 * TBS[0, 249] and st["bits_expired"] == 0


def test_conservation_random():
    rng = np.random.default_rng(1)
    n = 3000
    slot = np.sort(rng.integers(0, 400, n))
    r = req(slot, rng.integers(0, 40, n), rng.integers(0, 200_000, n))
    r["mcs"] = rng.integers(0, 28, n).astype(np.int8)
    g, used, st = schedule(r, 400, TBS, 250, 4)
    assert st["bits_served"] + st["bits_expired"] + st["bits_left_over"] == st["bits_requested"]
    assert (used <= 250).all() and (g.nprb >= 1).all() and (g.served <= g.tbs).all()
    per_slot = pd.Series(g.nprb).groupby(g.slot).sum()
    np.testing.assert_array_equal(per_slot.to_numpy(), used[per_slot.index])
    assert ((g.slot - r["slot"][g.oldest]) < 4).all()       # no grant carries bits older than 4 slots


def test_no_grants():
    g, used, st = schedule(req([], [], []), 5, TBS, 250, 4)
    assert len(g) == 0 and st["bits_requested"] == 0


# ---- requests and reception --------------------------------------------------------------

def fake_radio(holes=()):
    snr_min = np.repeat(np.arange(3, 31, dtype=np.int8)[:28, None], 250, axis=1)   # MCS m from 3 + m dB
    return Radio("x", TBS, np.zeros((28, 250, 3), np.int64), snr_min, 30, np.array(holes, np.int16).reshape(-1, 3),
                 np.arange(29, dtype=np.int8).clip(0, 26), np.arange(29, dtype=np.int8).clip(1, 27), np.full(29, 0.5))


def test_requests_slots():
    rows = {"t_ms": np.array([0, 0, 1, 5, 5], np.int32), "RNTI": np.array([1, 2, 1, 3, 1], np.uint16),
            "MCS": np.array([0, 3, 10, 28, 20], np.int8), "TBS": np.full(5, 100, np.int32),
            "bits": np.full(5, 100, np.int64), "src_block": np.zeros(5, np.int64)}
    r = rngs(0)
    q = requests(rows, fake_radio(), 2, r["mcs"], r["slot"])
    assert (np.diff(q["slot"]) >= 0).all()
    assert set(q["slot"]) <= {0, 1, 2, 3, 10, 11}
    assert "snr_db" not in q


def test_draw_snr_truncated_geometric_and_holes():
    """ADR-0020: snr_min + k, P(k) ∝ exp(-k / decay), capped at snr_max, never on a hole."""
    n = 200_000
    mcs, nprb = np.full(n, 20, np.int64), np.full(n, 7, np.int64)       # snr_min = 23, so k in 0..7
    rad = fake_radio(holes=[(20, 7, 24)])
    snr = draw_snr(mcs, nprb, rad, 1.5, np.random.default_rng(0)).astype(int)
    assert snr.min() == 23 and snr.max() <= 30 and not (snr == 24).any()
    k = np.array([0, 2, 3, 4, 5, 6, 7])
    p = np.exp(-k / 1.5); p /= p.sum()
    freq = np.array([(snr == 23 + i).mean() for i in k])
    np.testing.assert_allclose(freq, p, atol=3e-3)
    assert (draw_snr(mcs[:10], nprb[:10], rad, 0.0, np.random.default_rng(0)) == 23).all()
    a = draw_snr(mcs[:50], nprb[:50], rad, 1.5, np.random.default_rng(1))
    np.testing.assert_array_equal(a, draw_snr(mcs[:50], nprb[:50], rad, 1.5, np.random.default_rng(1)))


def test_reception_schema_and_order():
    g0 = Grants(*[np.array(v, np.int64) for v in ([0, 3], [10, 20], [5, 6], [100, 200], [100, 200], [0, 1], [0, 1], [1, 2])])
    g1 = Grants(*[np.array(v, np.int64) for v in ([1], [30], [7], [300], [300], [0], [0], [9])])
    jobs, tasks = reception([g0, g1], fake_radio(), TrafficConfig(), np.random.default_rng(0))
    st = TaskStream(jobs, tasks)
    st.validate()
    assert list(jobs.t_arrival_ms) == [2.5, 3.0, 4.0] and list(jobs.du) == [0, 1, 0]
    assert list(tasks.nprb) == [10, 10, 30, 30, 20, 20]
    assert (tasks.snr_db.to_numpy()[::2] == tasks.snr_db.to_numpy()[1::2]).all()        # same SNR on both tasks
    assert (tasks.snr_db >= 3 + tasks.mcs).all() and (tasks.snr_db <= 30).all()
    assert tasks.depends_on.isna().tolist() == [True, False] * 3
    assert tasks.depends_on[1::2].tolist() == tasks.task_id[::2].tolist()


# ---- cache -------------------------------------------------------------------------------

def test_write_once_is_atomic(tmp_path):
    calls = []

    def write(d):
        calls.append(d)
        (d / "x.txt").write_text("1")
        (d / cache.MANIFEST).write_text("{}")

    d = cache.write_once(tmp_path / "k", write)
    cache.write_once(tmp_path / "k", write)
    assert len(calls) == 1 and (d / "x.txt").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".tmp-")]


def test_file_hash_memo(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    h1 = cache.file_sha256(f, tmp_path)
    assert cache.file_sha256(f, tmp_path) == h1 == cache.file_sha256(f)
    f.write_bytes(b"abcd")
    assert cache.file_sha256(f, tmp_path) != h1
