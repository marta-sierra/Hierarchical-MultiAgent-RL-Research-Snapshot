"""Validation tests on the real trace (docs/TRAFFIC.md §12)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from acmad.traffic.api import generate, generate_from
from acmad.nr import MCS_T2
from acmad.traffic.radio import build_radio
from acmad.traffic.stream import TaskStream
from acmad.traffic.traces import build_prep

pytestmark = pytest.mark.trace


@pytest.fixture(scope="module")
def streams(cfg, prep, radio):
    cases = [("train", 1, 1.0, 0), ("train", 3, 4.0, 0), ("val", 6, 8.0, 1), ("test", 2, 16.0, 2)]
    return {c: generate_from(prep, radio, cfg, *c) for c in cases}


def test_validate(streams):
    for st in streams.values():
        st.validate()


def test_determinism(cfg, streams):
    for case, st in streams.items():
        again = generate(cfg, *case)
        assert again.content_hash() == st.content_hash()
        pd.testing.assert_frame_equal(again.jobs, st.jobs)
        pd.testing.assert_frame_equal(again.tasks, st.tasks)
    a, b = generate(cfg, "train", 3, 4.0, 0), generate(cfg, "train", 3, 4.0, 1)
    assert a.content_hash() != b.content_hash()


def test_cache_equals_fresh(cfg, prep, radio):
    """Stages built in memory from the CSVs == stages loaded from the cache (memory-mapped)."""
    fresh_prep = build_prep(cfg)
    fresh_radio = build_radio(cfg, fresh_prep)
    for case in [("train", 3, 4.0, 0), ("test", 6, 8.0, 3)]:
        a = generate_from(fresh_prep, fresh_radio, cfg, *case)
        b = generate_from(prep, radio, cfg, *case)
        pd.testing.assert_frame_equal(a.jobs, b.jobs)
        pd.testing.assert_frame_equal(a.tasks, b.tasks)


def test_save_load_roundtrip(streams, tmp_path):
    st = streams[("train", 3, 4.0, 0)]
    st.save(tmp_path / "s")
    back = TaskStream.load(tmp_path / "s")
    back.validate()
    pd.testing.assert_frame_equal(back.jobs, st.jobs)
    pd.testing.assert_frame_equal(back.tasks, st.tasks)
    assert back.content_hash() == st.content_hash()


def test_split_isolation(prep, streams):
    split_of = prep.blocks.set_index("block")["split"]
    for (split, *_), st in streams.items():
        assert {split_of[b] for b in st.meta["block_of_du"]} == {split}
        assert prep.superblocks.set_index("superblock").loc[st.meta["superblock"], "split"] == split
    sb = prep.superblocks
    assert sb.split.value_counts().to_dict() == {"train": 33, "val": 11, "test": 11}


def test_load_classes(prep):
    """ADR-0007: 20 / 15 / 20 super-blocks, ordered by load, each class split 3 : 1 : 1."""
    sb = prep.superblocks
    assert sb.load_class.value_counts().to_dict() == {"low": 20, "medium": 15, "high": 20}
    lo, me, hi = (sb.loc[sb.load_class == c, "mbps"] for c in ("low", "medium", "high"))
    assert lo.max() < me.min() and me.max() < hi.min()
    counts = sb.groupby(["load_class", "split"]).size().unstack()
    assert (counts["train"] == 3 * counts["val"]).all() and (counts["val"] == counts["test"]).all()
    assert (prep.blocks.merge(sb, on="superblock").eval("load_class_x == load_class_y")).all()


@pytest.mark.parametrize("load_class", ["low", "medium", "high"])
def test_generate_with_load_class(cfg, prep, radio, load_class):
    sb = prep.superblocks.set_index("superblock")
    drawn = set()
    for seed in range(10):
        st = generate_from(prep, radio, cfg, "train", 2, 1.0, seed, load_class)
        m = st.meta
        assert m["load_class"] == load_class and m["superblock_load_class"] == load_class
        assert sb.at[m["superblock"], "split"] == "train" and sb.at[m["superblock"], "load_class"] == load_class
        drawn.add(m["superblock"])
    assert len(drawn) > 1


def test_load_class_none_draws_any(cfg, prep, radio, streams):
    """None = every class of the split: the same stream as without the argument."""
    case = ("train", 3, 4.0, 0)
    st = generate_from(prep, radio, cfg, *case, None)
    assert st.content_hash() == streams[case].content_hash() and st.meta["load_class"] is None
    seen = {generate_from(prep, radio, cfg, "train", 1, 1.0, seed).meta["superblock_load_class"] for seed in range(30)}
    assert seen == {"low", "medium", "high"}
    with pytest.raises(ValueError, match="load_class"):
        generate_from(prep, radio, cfg, "train", 1, 1.0, 0, "extreme")


def test_split_is_independent_of_episode_seed(cfg, prep):
    a = [generate(cfg, "val", 1, 1.0, seed).meta["superblock"] for seed in range(20)]
    val = set(prep.superblocks.loc[prep.superblocks.split == "val", "superblock"])
    assert set(a) <= val and len(set(a)) > 1


def test_conservation(streams):
    for st in streams.values():
        m = st.meta
        assert m["bits_requested"] == m["bits_served"] + m["bits_expired"] + m["bits_left_over"]
        for d in m["per_du"]:
            assert d["bits_requested"] == d["bits_served"] + d["bits_expired"] + d["bits_left_over"]
        t = st.tasks
        assert len(t) == 2 * len(st.jobs) and m["granted"] == len(st.jobs)
        cpu, ldpc = t.iloc[::2], t.iloc[1::2]
        assert (cpu.kind == "cpu_processing").all() and (ldpc.kind == "ldpc_decoding").all()
        assert (cpu.allowed == "cpu").all() and (ldpc.allowed == "any").all()
        assert cpu.depends_on.isna().all()
        np.testing.assert_array_equal(ldpc.depends_on.to_numpy(np.int64), cpu.task_id.to_numpy())
        np.testing.assert_array_equal(cpu.job_id.to_numpy(), ldpc.job_id.to_numpy())
        for c in ("nprb", "mcs", "snr_db", "tbs"):
            np.testing.assert_array_equal(cpu[c].to_numpy(), ldpc[c].to_numpy())
        assert not t[["nprb", "mcs", "snr_db", "tbs"]].isna().any().any()


def test_timing(cfg, streams):
    off = cfg.timing.fronthaul_ms + cfg.timing.jitter_ms
    for st in streams.values():
        t = st.jobs.t_arrival_ms.to_numpy() - off
        assert (np.mod(t, cfg.radio.slot_ms) == 0).all()
        first = (cfg.timing.k2_slots + 1) * cfg.radio.slot_ms
        assert t.min() >= first and t.max() < cfg.split.block_ms + first


def grant_slots(cfg, st):
    t = st.jobs.t_arrival_ms.to_numpy() - cfg.timing.fronthaul_ms - cfg.timing.jitter_ms
    return np.rint(t / cfg.radio.slot_ms).astype(np.int64) - cfg.timing.k2_slots - 1


def test_radio_limits(cfg, prep, streams):
    for st in streams.values():
        cpu = st.tasks.iloc[::2]
        assert cpu.nprb.min() >= 1 and cpu.nprb.max() <= cfg.radio.pusch_prb
        per = pd.Series(cpu.nprb.to_numpy(np.int64)).groupby([st.jobs.du.to_numpy(), grant_slots(cfg, st)]).sum()
        assert per.max() <= cfg.radio.pusch_prb
        assert set(st.jobs.du.unique()) <= set(range(st.meta["n_dus"]))
    assert not np.isin(np.asarray(prep.rows["RNTI"]), cfg.trace.drop_rntis).any()
    assert (np.asarray(prep.rows["MCS"]) <= 28).all()


def test_tbs_and_code_blocks_match_cloudric(radio, cloudric):
    for f, d in cloudric.items():
        d = d[["nPRB", "MCS", "TBS", "Base_Graph", "Codeblock_Length", "Number_Codeblocks", "Total_Bits"]].drop_duplicates()
        m, n = d.MCS.to_numpy(), d.nPRB.to_numpy() - 1
        np.testing.assert_array_equal(radio.tbs[m, n], d.TBS, err_msg=f)
        np.testing.assert_array_equal(radio.cb[m, n, 0], d.Base_Graph, err_msg=f)
        np.testing.assert_array_equal(radio.cb[m, n, 1], d.Number_Codeblocks, err_msg=f)
        np.testing.assert_array_equal(radio.cb[m, n, 2], d.Codeblock_Length, err_msg=f)
        np.testing.assert_array_equal(radio.cb[m, n, 1] * radio.cb[m, n, 2], d.Total_Bits, err_msg=f)


def test_measured_domain(streams, cloudric):
    """ADR-0016, ADR-0020: every (nPRB, SNR, MCS) of a task was measured (decoded) on both CloudRIC decoders."""
    ok = set.intersection(*(set(zip(d.nPRB.astype(int), d.SNR_dB.astype(int), d.MCS.astype(int)))
                            for d in cloudric.values()))
    for st in streams.values():
        t = st.tasks
        triples = set(zip(t.nprb.astype(int), t.snr_db.astype(int), t.mcs.astype(int)))
        assert triples <= ok, sorted(triples - ok)[:10]


def test_ldpc_columns_match_cloudric(streams, cloudric):
    """ADR-0019: base_graph, n_cb, cb_len, total_bits, qm, code_rate follow (mcs, nprb) as in CloudRIC's data."""
    ref = cloudric["CPU"][["nPRB", "MCS", "TBS", "Base_Graph", "Number_Codeblocks", "Codeblock_Length", "Total_Bits"]]
    ref = ref.drop_duplicates().rename(columns={"nPRB": "nprb", "MCS": "mcs"}).set_index(["nprb", "mcs"])
    for st in streams.values():
        t = st.tasks.drop_duplicates(["nprb", "mcs"]).set_index(["nprb", "mcs"])
        r = ref.loc[t.index]
        np.testing.assert_array_equal(t.tbs, r.TBS)
        np.testing.assert_array_equal(t.base_graph, r.Base_Graph)
        np.testing.assert_array_equal(t.n_cb, r.Number_Codeblocks)
        np.testing.assert_array_equal(t.cb_len, r.Codeblock_Length)
        np.testing.assert_array_equal(t.total_bits, r.Total_Bits)
        qm = np.array([q for q, _ in MCS_T2])[t.index.get_level_values("mcs")]
        np.testing.assert_array_equal(t.qm, qm)
        assert (st.tasks.code_rate > 0).all() and (st.tasks.code_rate < 1).all()


def test_snr_table(radio, cloudric):
    """ADR-0020: snr_min is the lowest SNR measured on both decoders; the table + holes rebuild every triple."""
    common = set.intersection(*(set(zip(d.MCS.astype(int), d.nPRB.astype(int), d.SNR_dB.astype(int)))
                                for d in cloudric.values()))
    holes = set(map(tuple, radio.snr_holes.tolist()))
    rebuilt = {(m, n + 1, x) for m in range(radio.snr_min.shape[0]) for n in range(radio.snr_min.shape[1])
               for x in range(int(radio.snr_min[m, n]), radio.snr_max + 1)} - holes
    assert rebuilt == common


def test_x1_calibration(prep, streams):
    st = streams[("train", 1, 1.0, 0)]
    src = prep.blocks.set_index("block").loc[st.meta["block_of_du"][0], "mbps"]
    assert st.meta["requested_mbps"] == pytest.approx(src, abs=1e-6)


def test_load_monotonicity(cfg, prep, radio):
    base = generate_from(prep, radio, cfg, "train", 2, 1.0, 5).meta["requested_mbps"]
    for s in (2.0, 4.0, 8.0):
        m = generate_from(prep, radio, cfg, "train", 2, s, 5).meta["requested_mbps"]
        assert m == pytest.approx(s * base, rel=1e-3)
    loads = [generate_from(prep, radio, cfg, "train", n, 1.0, 5).meta["requested_mbps"] for n in (1, 2, 4, 6)]
    assert all(b > a for a, b in zip(loads, loads[1:]))


def test_meta_contents(streams):
    m = streams[("train", 3, 4.0, 0)].meta
    for k in ("config", "config_hash", "generator_version", "git_hash", "seed", "seed_layout", "split", "superblock",
              "block_of_du", "n_dus", "s", "T_ms", "slot_ms", "carrier_prb", "slot_prb", "k2_slots", "re_per_prb",
              "layers", "fronthaul_ms", "jitter_ms", "requested_mbps", "granted_tbs_mbps", "prb_util_mean",
              "prb_util_p99", "jobs_per_s", "bits_requested", "bits_served", "bits_expired", "bits_left_over",
              "wait_ms_p99", "per_du", "prep_key", "radio_key"):
        assert k in m, k
    assert len(m["per_du"]) == 3 and m["seed_layout"]["spawn"] == ["scenario", "snr", "slot", "mcs"]


def test_performance(cfg):
    generate(cfg, "train", 6, 8.0, 0)                  # warm
    t0 = time.perf_counter()
    generate(cfg, "train", 6, 8.0, 0)
    assert time.perf_counter() - t0 < 1.0              # target 0.4 s; loose bound for slow machines
