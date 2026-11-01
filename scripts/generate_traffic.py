"""Build stages A (trace prep + split) and R (radio tables), then the fixed streams.

    .venv/bin/python scripts/generate_traffic.py [--config configs/traffic/default.yaml]
        [--case val,3,4.0,0[,high] ...]
        [--splits val test --n-dus 1 3 6 --s 1 4 8 --seeds 0 1 2 --load-classes any low high]

Streams go to `data/cache/traffic/<hash>/` (docs/TRAFFIC.md §11.2); an existing stream is
not regenerated. Without `--case`, the grid `splits x n_dus x s x seeds x load classes` is
generated. `any` (the default) draws the super-block from every load class (ADR-0007).
"""
from __future__ import annotations

import argparse
import itertools
import time

from acmad.traffic import LOAD_CLASSES, load_config, load_or_generate, stream_dir
from acmad.traffic.config import DEFAULT_CONFIG
from acmad.traffic.radio import ensure_radio
from acmad.traffic.traces import ensure_prep, prep_dir


def parse_load_class(text: str) -> str | None:
    if text != "any" and text not in LOAD_CLASSES:
        raise argparse.ArgumentTypeError(f"load class must be 'any' or one of {LOAD_CLASSES}")
    return None if text == "any" else text


def parse_case(text: str) -> tuple[str, int, float, int, str | None]:
    split, n_dus, s, seed, *load_class = text.split(",")
    return split, int(n_dus), float(s), int(seed), parse_load_class(load_class[0] if load_class else "any")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--case", action="append", type=parse_case, default=[], help="split,n_dus,s,seed[,load_class]")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--n-dus", nargs="+", type=int, default=[1, 3, 6])
    ap.add_argument("--s", nargs="+", type=float, default=[1.0, 4.0, 8.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--load-classes", nargs="+", type=parse_load_class, default=[None],
                    help=f"any | {' | '.join(LOAD_CLASSES)}")
    ap.add_argument("--prep-only", action="store_true", help="build stages A and R only")
    a = ap.parse_args()

    cfg = load_config(a.config)
    t0 = time.perf_counter()
    prep = ensure_prep(cfg)
    radio = ensure_radio(cfg, prep)
    print(f"stage A {prep_dir(cfg)}\nstage R key {radio.key}  ({time.perf_counter() - t0:.1f} s)")
    if a.prep_only:
        return
    cases = a.case or list(itertools.product(a.splits, a.n_dus, a.s, a.seeds, a.load_classes))
    for case in cases:
        t0 = time.perf_counter()
        st = load_or_generate(cfg, *case)
        print(f"{case}: {len(st.jobs):>7} jobs, {st.meta['requested_mbps']:7.1f} Mb/s requested, "
              f"{time.perf_counter() - t0:5.2f} s -> {stream_dir(cfg, *case)}")


if __name__ == "__main__":
    main()
