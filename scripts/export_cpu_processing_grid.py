"""Write the configurations the MATLAB `cpu_processing` measurement runs (ADR-0008).

    .venv/bin/python scripts/export_cpu_processing_grid.py [--out measurements/cpu_processing/grid.csv]

One row per (MCS, nPRB): every MCS of table 2 x the nPRB values below, with the sizes our
generator gives them (`acmad.nr`) so MATLAB can assert it builds the same transport block, and
the SNR at which the LDPC time is compared with CloudRIC (the lowest SNR CloudRIC measured for
that (MCS, nPRB) + 1 dB, ADR-0020).
"""
from __future__ import annotations

import argparse

import pandas as pd

from acmad.nr import MCS_T2, N_MCS
from acmad.traffic import load_config
from acmad.traffic.config import REPO_ROOT
from acmad.traffic.radio import ensure_radio

NPRB = (1, 2, 4, 8, 12, 16, 25, 32, 50, 64, 80, 100, 128, 150, 175, 200, 225, 250)
SNR_ABOVE_MIN_DB = 1


def grid() -> pd.DataFrame:
    cfg = load_config()
    radio = ensure_radio(cfg)
    rows = []
    for m in range(N_MCS):
        qm, r = MCS_T2[m]
        for n in NPRB:
            bg, n_cb, cb_len = (int(x) for x in radio.cb[m, n - 1])
            rows.append({"mcs": m, "qm": qm, "r_x1024": r, "nprb": n, "tbs": int(radio.tbs[m, n - 1]),
                         "base_graph": bg, "n_cb": n_cb, "cb_len": cb_len, "total_bits": n_cb * cb_len,
                         "g_bits": n * cfg.radio.re_per_prb * qm * cfg.radio.layers,
                         "snr_db": min(int(radio.snr_min[m, n - 1]) + SNR_ABOVE_MIN_DB, radio.snr_max)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "measurements" / "cpu_processing" / "grid.csv"))
    a = ap.parse_args()
    g = grid()
    g.to_csv(a.out, index=False)
    print(f"{len(g)} configurations -> {a.out}")


if __name__ == "__main__":
    main()
