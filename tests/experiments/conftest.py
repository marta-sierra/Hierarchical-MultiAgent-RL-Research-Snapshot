"""Experiment tests reuse the simulator test helpers (tests/sim/simhelpers.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
