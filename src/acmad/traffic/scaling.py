"""Stage B0: episode. One super-block of the split (and of the load class, if given), one
block per DU (least used, random ties), bits x s (docs/TRAFFIC.md §5; ADR-0017, ADR-0007)."""
from __future__ import annotations

import numpy as np

from .traces import Prep


def assign_blocks(cand: np.ndarray, n_dus: int, rng: np.random.Generator) -> list[int]:
    """Each DU takes the least-used block so far; ties broken by a uniform draw per candidate."""
    uses = np.zeros(len(cand))
    per_du = []
    for _ in range(n_dus):
        pick = int(np.argmin(uses + rng.random(len(cand))))
        uses[pick] += 1
        per_du.append(int(cand[pick]))
    return per_du


def build_episode(prep: Prep, split: str, n_dus: int, s: float, rng: np.random.Generator,
                  load_class: str | None = None) -> tuple[list[dict[str, np.ndarray]], dict]:
    """Rows of each DU (times from the block start, `bits = round(TBS * s)`) and the scenario meta.

    Uses the `scenario` stream: one draw of the super-block, then one uniform per candidate
    block per DU (prototype order, kept for reproducibility). `load_class=None` draws from
    every class of the split.
    """
    sb_table = prep.superblocks
    pick = sb_table["split"] == split
    if load_class is not None:
        pick &= sb_table["load_class"] == load_class
    sbs = sb_table.loc[pick, "superblock"]
    if len(sbs) == 0:
        raise ValueError(f"no super-block in split {split!r}" + ("" if load_class is None else f", load class {load_class!r}"))
    sb = int(rng.choice(sbs))
    cand = prep.blocks.loc[prep.blocks["superblock"] == sb, "block"].to_numpy()
    block_of_du = assign_blocks(cand, n_dus, rng)
    per_du = []
    for b in block_of_du:
        rows = prep.block_rows(b)                      # already sorted by t_ms inside the block
        rows["bits"] = np.rint(rows["TBS"] * s).astype(np.int64)
        rows["src_block"] = np.full(len(rows["TBS"]), b, dtype=np.int64)
        per_du.append(rows)
    sb_load_class = str(sb_table.set_index("superblock").at[sb, "load_class"])
    return per_du, {"split": split, "load_class": load_class, "superblock": sb,
                    "superblock_load_class": sb_load_class, "block_of_du": block_of_du, "n_dus": n_dus, "s": s}
