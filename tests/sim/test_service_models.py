"""ldpc_decoding service models on CloudRIC's data (ADR-0003, ADR-0004). Skip without data/cloudric/."""
from __future__ import annotations

import numpy as np
import pytest

from acmad.cloudric import DEFAULT_DIR, read_cloudric
from acmad.nr import MCS_T2, code_blocks, tbs_bits
from acmad.sim.models import ExecutorSpec, TaskFeatures, make_model

pytestmark = pytest.mark.skipif(not (DEFAULT_DIR / "CPU_dataset.csv").exists(), reason="data/cloudric/ not available")


def ldpc(mcs: int, nprb: int, snr: int) -> TaskFeatures:
    tbs = tbs_bits(mcs, nprb)
    bg, n_cb, cb_len = code_blocks(tbs, mcs)
    qm, r = MCS_T2[mcs]
    return TaskFeatures("ldpc_decoding", nprb, mcs, snr, tbs, bg, n_cb, cb_len, n_cb * cb_len, qm, r / 1024)


@pytest.fixture(scope="module")
def tables():
    from acmad.sim.models.cloudric_ldpc import load_tables
    return load_tables()


@pytest.fixture(scope="module")
def cloudric():
    return read_cloudric(DEFAULT_DIR)


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_draws_centre_on_the_measured_median(kind, cloudric, tables):
    d = cloudric[kind.upper()]
    for mcs, nprb, snr in [(20, 250, 30), (1, 5, 6), (10, 100, 14)]:
        runs = d[(d.MCS == mcs) & (d.nPRB == nprb) & (d.SNR_dB == snr)]
        f = ldpc(mcs, nprb, snr)
        x = np.array([make_model(kind)(f, ExecutorSpec(kind), np.random.default_rng(i)) for i in range(3000)])
        assert np.median(x[:, 0]) == pytest.approx(runs.Decoding_Time_us.median() * 1e-3, rel=0.01)
        e_tot = x[:, 1] + tables[kind].static_power_w * x[:, 0] * 1e-3
        assert np.median(e_tot) == pytest.approx(runs.Decoding_Energy_uJ.median() * 1e-6, rel=0.01)
        assert (x[:, 0] > 0).all() and (x[:, 1] >= 0).all()


def test_static_power(tables, cloudric):
    assert tables["cpu"].static_power_w == 0.0
    assert tables["gpu"].static_power_w == pytest.approx(np.quantile(cloudric["GPU"].Decoding_Power_W, 0.01))


def test_heavy_tail_is_kept(tables):
    """ADR-0003: the pooled residuals keep CloudRIC's rare large slowdowns (CPU max ~x33)."""
    assert max(np.exp(r[:, 0].max()) for r in tables["cpu"].res) > 30


def test_every_generated_ldpc_task_is_in_the_tables(tables):
    from acmad.traffic import generate, load_config
    cfg = load_config()
    if not cfg.trace_path.exists():
        pytest.skip("trace not available")
    t = generate(cfg, "val", 3, 8.0, 0).tasks
    t = t[t.kind == "ldpc_decoding"]
    for k in ("cpu", "gpu"):
        v = tables[k].t_ms[t.mcs.to_numpy(), t.nprb.to_numpy() - 1, t.snr_db.to_numpy()]
        assert not np.isnan(v).any()


def test_gpu_speed_other_than_one_is_deferred():
    with pytest.raises(NotImplementedError, match="ADR-0010"):
        make_model("gpu")(ldpc(10, 50, 20), ExecutorSpec("gpu", params={"speed": 2.0}), np.random.default_rng(0))


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_closed_form_mean_matches_the_draws(kind):
    """ADR-0009 §2: mean_service_ms equals the Monte Carlo mean of the model's draws."""
    import pandas as pd
    feats = [ldpc(20, 250, 30), ldpc(1, 5, 6), ldpc(10, 100, 14)]
    if kind == "cpu":
        f = ldpc(10, 100, 14)
        feats.append(TaskFeatures("cpu_processing", *[getattr(f, n) for n in list(TaskFeatures.__dataclass_fields__)[1:]]))
    model, spec = make_model(kind), ExecutorSpec(kind)
    from dataclasses import asdict
    tasks = pd.DataFrame([asdict(f) for f in feats])
    mean = model.mean_service_ms(tasks, spec)
    for f, m in zip(feats, mean):
        x = np.array([model(f, spec, np.random.default_rng(i))[0] for i in range(20000)])
        assert m == pytest.approx(x.mean(), rel=0.03)
    if kind == "gpu":
        cp = tasks.assign(kind="cpu_processing")
        assert np.isnan(model.mean_service_ms(cp, spec)).all()
