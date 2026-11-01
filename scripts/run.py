"""Run an experiment (configs/experiment/*.yaml) and write its report.

    .venv/bin/python scripts/run.py configs/experiment/benchmark.yaml [--workers 32] [--out results]
        [--resume results/benchmark/<run>] [--baseline rr]

Every (method, scenario, seed) cell runs in its own process; the run folder layout is in
`acmad.experiments.runner`. `--resume` reruns only the cells without a metrics.json (same config).
"""
from __future__ import annotations

import argparse
import sys

from acmad.experiments.analysis import report
from acmad.experiments.config import REPO, load_experiment
from acmad.experiments.runner import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=None, help="parallel processes (default: all cores)")
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--resume", default=None, help="run folder to complete")
    ap.add_argument("--baseline", default=None, help="method for the paired differences in report.md")
    a = ap.parse_args()
    cfg = load_experiment(a.config)
    try:
        run_dir = run_experiment(cfg, a.out, a.workers, a.resume)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)
    (run_dir / "report.md").write_text(report(run_dir, baseline=a.baseline))
    print(f"results: {run_dir}")


if __name__ == "__main__":
    main()
