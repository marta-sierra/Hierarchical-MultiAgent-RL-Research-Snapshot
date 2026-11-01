"""The mechanics against queueing theory: M/M/1, M/M/c and M/D/1 waits in queue (docs/SIM_VALIDATION.md)."""
import pytest

from acmad.experiments.validation import QueueCase, erlang_c, run_queue, theory

N_JOBS = 40_000


def test_theory_formulas():
    assert erlang_c(1, 0.7) == pytest.approx(0.7)                       # M/M/1: P(wait) = ρ
    assert theory("exp", 1, 0.5, 0.7)["wq_ms"] == pytest.approx(0.7 * 0.5 / 0.3)   # ρ/(μ-λ)
    assert theory("det", 1, 0.5, 0.7)["wq_ms"] == pytest.approx(theory("exp", 1, 0.5, 0.7)["wq_ms"] / 2)
    assert erlang_c(2, 1.0) == pytest.approx(1 / 3)                     # textbook value


@pytest.mark.parametrize("case", [
    QueueCase("gpu", "exp", 1, 0.5, 0.7),      # M/M/1 on the GPU (ldpc_decoding)
    QueueCase("gpu", "exp", 3, 0.5, 0.8),      # M/M/3: capacity-c FIFO
    QueueCase("gpu", "det", 1, 0.5, 0.7),      # M/D/1: Pollaczek-Khinchine
    QueueCase("cpu", "exp", 2, 0.3, 0.75),     # M/M/2 on the CPU (cpu_processing, rule-routed)
], ids=lambda c: c.label)
def test_queue_matches_theory(case):
    r = run_queue(case, N_JOBS, seed=0)
    assert abs(r["wq_ms"] - r["theory_wq_ms"]) < 4 * r["wq_se_ms"], r          # within 4 standard errors
    assert abs(r["wq_ms"] / r["theory_wq_ms"] - 1) < 0.15, r
    assert r["p_wait"] == pytest.approx(r["theory_p_wait"], abs=0.02)
    assert r["util"] == pytest.approx(case.rho, abs=0.02)
