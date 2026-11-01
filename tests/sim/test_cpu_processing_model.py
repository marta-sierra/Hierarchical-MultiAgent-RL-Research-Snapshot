"""`cpu_processing` model from the MATLAB measurement (ADR-0008), on synthetic measurements."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acmad.sim.models import matlab_cpu_processing as M
from acmad.sim.models.base import TaskFeatures
from acmad.sim.models.cloudric_ldpc import LdpcTables

COEF = np.array([0.6, 0.01, 1e-5, 4e-6])           # a [ms], per PRB, per bit of G, per bit of total_bits
K = 20.0


def synthetic(iters: int = 10, n_rep: int = 30, seed: int = 0) -> tuple[pd.DataFrame, LdpcTables]:
    """times.csv-like rows whose block sum follows COEF with ±2 % noise; MATLAB LDPC = K x CloudRIC."""
    rng = np.random.default_rng(seed)
    rows = []
    for mcs, qm, rate in ((0, 2, 0.12), (4, 2, 0.59), (5, 4, 0.37), (10, 4, 0.64), (15, 6, 0.65), (25, 8, 0.86)):
        for nprb in (1, 8, 50, 100, 250):
            g = nprb * 156 * qm
            tb = round(g * rate)                       # total_bits ~ G x code rate: not collinear with G
            t = COEF @ [1, nprb, g, tb] * np.exp(rng.normal(0, 0.02, n_rep))
            ldpc = K * (0.01 + 1e-5 * tb)
            rows.append(pd.DataFrame({"mcs": mcs, "nprb": nprb, "g_bits": g, "total_bits": tb, "snr_db": 20,
                                      "t_chest_ms": t * 0.4, "t_extract_ms": t * 0.1, "t_eq_ms": t * 0.1,
                                      "t_demod_ms": t * 0.3, "t_ratercv_ms": t * 0.1,
                                      "t_ldpc_ms": np.r_[np.full(5, ldpc), np.full(n_rep - 5, np.nan)],
                                      "ldpc_iters": np.r_[np.full(5, iters), np.full(n_rep - 5, np.nan)]}))
    times = pd.concat(rows, ignore_index=True)
    t_ms = np.full((28, 250, 31), np.nan)
    for (mcs, nprb, tb), _ in times.groupby(["mcs", "nprb", "total_bits"]):
        t_ms[mcs, nprb - 1, 20] = 0.01 + 1e-5 * tb
    return times, LdpcTables("cpu", t_ms, t_ms, np.array([]), (), 0.0)


def task(nprb: int, qm: int) -> TaskFeatures:
    return TaskFeatures("cpu_processing", nprb, 10, 20, 1000, 1, 1, 100 * qm, nprb * 100 * qm, qm, 0.5)


def test_fit_recovers_the_formula():
    times, _ = synthetic()
    coef, res = M.fit_formula(times)
    cfg = times.groupby(["mcs", "nprb"]).first()
    X = np.c_[np.ones(len(cfg)), cfg.index.get_level_values("nprb"), cfg.g_bits, cfg.total_bits]
    assert X @ coef == pytest.approx(X @ COEF, rel=0.01)       # predicted medians
    assert coef[0] == pytest.approx(COEF[0], rel=0.05)          # MATLAB's fixed cost a
    assert abs(np.median(res)) < 0.01 and np.std(res) == pytest.approx(0.02, rel=0.2)


def test_k_is_the_ldpc_ratio_and_needs_fixed_iterations():
    times, ldpc = synthetic()
    assert M.calibration_k(times, ldpc) == pytest.approx(K)
    early, ldpc = synthetic(iters=4)
    with pytest.raises(ValueError, match="fixed 10 iterations"):
        M.calibration_k(early, ldpc)


def test_server_time_is_the_formula_divided_by_k():
    model = M.build(*synthetic())
    assert model.keep_constant                                   # the author's choice (ADR-0008)
    a, b, c, d = model.coef
    for nprb, qm in ((1, 2), (250, 8)):
        g, tb = nprb * 156 * qm, nprb * 100 * qm         # total_bits of task(nprb, qm)
        assert model.median_ms(nprb, g, tb) == pytest.approx((a + b * nprb + c * g + d * tb) / model.k)
        draws = [model.draw(task(nprb, qm), np.random.default_rng(i)) for i in range(2000)]
        assert np.median(draws) == pytest.approx(model.median_ms(nprb, g, tb), rel=0.01)
    without = M.CpuProcessingModel(model.coef, model.k, model.residuals, keep_constant=False)
    assert model.median_ms(1, 312, 200) - without.median_ms(1, 312, 200) == pytest.approx(a / model.k)


def test_draw_is_reproducible_with_the_task_rng():
    model = M.build(*synthetic())
    t = task(50, 6)
    assert model.draw(t, np.random.default_rng(3)) == model.draw(t, np.random.default_rng(3))


@pytest.mark.skipif(not (M.RESULTS_DIR / M.RUN / "times.csv").exists(), reason="MATLAB run not available")
def test_real_run_gives_a_plausible_model():
    """Run v3 (ADR-0008): fixed 10 LDPC iterations, k in a sane range, cpu_processing well below
    CloudRIC's LDPC time for a large grant, and the shipped cpu model uses it."""
    from acmad.sim.models import ExecutorSpec, make_model
    model = M.load()
    assert 10 < model.k < 100
    assert (model.coef > 0).all()
    f = task(250, 8)
    t = make_model("cpu")(f, ExecutorSpec("cpu"), np.random.default_rng(0))
    assert t[1] == 0.0 and 0.05 < t[0] < 1.0                  # ms; LDPC of such a grant is ~3 ms
