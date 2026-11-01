"""`cpu_processing` service time from the MATLAB per-block measurement (ADR-0008).

    t(x) = t_MATLAB(x) / k * exp(eps)              (KEEP_CONSTANT; otherwise (t_MATLAB(x) - a) / k)
    t_MATLAB(x) = a + b * nPRB + c * G + d * total_bits

- `t_MATLAB`: least-squares fit (relative error) of the per-configuration medians of blocks 2-6
  (channel estimation, extraction, MMSE, demodulation, rate recovery); G = 156 * nPRB * Qm coded
  bits, total_bits = n_cb * cb_len.
- `a`: fixed cost per grant (mostly channel estimation), scaled by k like the rest (ADR-0008,
  amendment 4): a / k = 21.7 us per task with run v3.
- `k`: median over the grid of MATLAB / CloudRIC (FlexRAN CPU) LDPC time at the same
  (MCS, nPRB, SNR). FlexRAN always runs 10 iterations, so the MATLAB run must too
  (`ldpcTermination = 'max'`); a run with early termination is refused.
- `eps`: option A, resampled from the measured log-residuals `log(t_rep / t_MATLAB(x))`.
- Energy: not modelled (0).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from acmad.sim.models.cloudric_ldpc import LdpcTables, load_tables

REPO = Path(__file__).resolve().parents[4]
RESULTS_DIR = REPO / "measurements" / "cpu_processing" / "results"
RUN = "Martas-MacBook-Air.local_20261008_v3"      # the measurement in use (results/<RUN>/times.csv)
BLOCKS = ("t_chest_ms", "t_extract_ms", "t_eq_ms", "t_demod_ms", "t_ratercv_ms")
FEATURES = ("nprb", "g_bits", "total_bits")       # columns of times.csv, in the order of (b, c, d)
RE_PER_PRB = 156                                  # 1 layer: G = nPRB * 156 * Qm, as in grid.csv
CLOUDRIC_LDPC_ITERS = 10                          # FlexRAN CPU: every CloudRIC run reports 10
KEEP_CONSTANT = True                              # ADR-0008 amendment 4 (author): keep a, scaled by k


@dataclass(frozen=True, eq=False)
class CpuProcessingModel:
    """Fitted coefficients `(a, b, c, d)` in MATLAB ms, calibration factor `k`, log-residual pool."""

    coef: np.ndarray
    k: float
    residuals: np.ndarray
    keep_constant: bool = KEEP_CONSTANT

    def median_ms(self, nprb: int, g_bits: int, total_bits: int) -> float:
        """Server-level time without noise."""
        a, b, c, d = self.coef
        var = b * nprb + c * g_bits + d * total_bits
        return float((var + (a if self.keep_constant else 0.0)) / self.k)

    def mean_ms(self, nprb: np.ndarray, qm: np.ndarray, total_bits: np.ndarray) -> np.ndarray:
        """Vectorised E[t]: the noiseless time times mean(exp(eps)) of the residual pool (ADR-0009 §2)."""
        a, b, c, d = self.coef
        t0 = (b * nprb + c * nprb * RE_PER_PRB * qm + d * total_bits + (a if self.keep_constant else 0.0)) / self.k
        return t0 * float(np.exp(self.residuals).mean())

    def draw(self, task, rng: np.random.Generator) -> float:
        """Service time of one `cpu_processing` task (`TaskFeatures`), ms."""
        t0 = self.median_ms(task.nprb, task.nprb * RE_PER_PRB * task.qm, task.total_bits)
        return t0 * float(np.exp(self.residuals[rng.integers(len(self.residuals))]))


def fit_formula(times: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(a, b, c, d)` fitted to the per-configuration medians of blocks 2-6 (weights 1/t: relative
    error), and the log-residuals of every repetition against it."""
    t = times[list(BLOCKS)].sum(axis=1)
    cfg = times.assign(t=t).groupby(["mcs", "nprb"]).agg(t=("t", "median"), **{f: (f, "first") for f in FEATURES})
    A = np.c_[np.ones(len(cfg)), cfg[list(FEATURES)].to_numpy(float)]
    w = 1.0 / cfg.t.to_numpy()
    coef = np.linalg.lstsq(A * w[:, None], cfg.t.to_numpy() * w, rcond=None)[0]
    if (coef[1:] <= 0).any():
        raise ValueError(f"fitted slopes must be positive, got {coef}")
    fitted = np.c_[np.ones(len(times)), times[list(FEATURES)].to_numpy(float)] @ coef
    return coef, np.log(t.to_numpy() / fitted)


def calibration_k(times: pd.DataFrame, ldpc: LdpcTables) -> float:
    """Median over the grid of MATLAB LDPC time / CloudRIC FlexRAN LDPC time, same (MCS, nPRB, SNR)."""
    l = times.dropna(subset=["t_ldpc_ms"])
    if not (l.ldpc_iters == CLOUDRIC_LDPC_ITERS).all():
        raise ValueError(f"the MATLAB LDPC must run a fixed {CLOUDRIC_LDPC_ITERS} iterations like CloudRIC's "
                         "FlexRAN (ldpcTermination = 'max'); this run used early termination")
    m = l.groupby(["mcs", "nprb", "snr_db"]).t_ldpc_ms.median().reset_index()
    cr = ldpc.t_ms[m.mcs.to_numpy(), m.nprb.to_numpy() - 1, m.snr_db.to_numpy()]
    if np.isnan(cr).any():
        raise ValueError("a grid point has no CloudRIC LDPC measurement")
    return float(np.median(m.t_ldpc_ms.to_numpy() / cr))


def build(times: pd.DataFrame, ldpc: LdpcTables) -> CpuProcessingModel:
    coef, res = fit_formula(times)
    return CpuProcessingModel(coef, calibration_k(times, ldpc), res)


@lru_cache(maxsize=1)
def load(run_dir: Path = RESULTS_DIR / RUN) -> CpuProcessingModel:
    """The model from `results/<RUN>/times.csv` (fitted on first use in the process)."""
    f = Path(run_dir) / "times.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run measurements/cpu_processing/runCpuProcessing.m (ADR-0008)")
    return build(pd.read_csv(f), load_tables()["cpu"])
