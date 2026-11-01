"""Print (and save as report.md) the comparison tables of a finished run.

    .venv/bin/python scripts/analyse.py results/benchmark/<run> [--baseline rr]
        [--metrics job_hit_rate energy_per_job_mj latency_ms_p99]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from acmad.experiments.analysis import DEFAULT_METRICS, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    a = ap.parse_args()
    text = report(a.run_dir, a.metrics, a.baseline)
    (Path(a.run_dir) / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
