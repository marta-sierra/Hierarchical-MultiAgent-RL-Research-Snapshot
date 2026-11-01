"""CloudRIC's measured LDPC decoders (`data/cloudric/`, ADR-0016, ADR-0005): the one reader of those files.

Shared by `acmad.traffic` (SNR table, ADR-0020) and `acmad.sim.models` (service models, ADR-0003).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CLOUDRIC_FILES = ("CPU", "GPU")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = REPO_ROOT / "data" / "cloudric"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"


def read_cloudric(cloudric_dir: Path) -> dict[str, pd.DataFrame]:
    """CloudRIC CPU and GPU decoder datasets (one corrupt energy value is coerced to NaN)."""
    cr = {f: pd.read_csv(cloudric_dir / f"{f}_dataset.csv", low_memory=False) for f in CLOUDRIC_FILES}
    cr["CPU"]["Decoding_Energy_uJ"] = pd.to_numeric(cr["CPU"]["Decoding_Energy_uJ"], errors="coerce")
    return cr
