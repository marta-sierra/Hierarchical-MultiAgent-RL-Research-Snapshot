"""NR PHY sizes, from (MCS, nPRB): MCS table 2, TBS (TS 38.214 §5.1.3.2), LDPC base graph and
code-block segmentation (TS 38.212 §5.2.2 / §7.2.2). The mapping is written out in ADR-0019.

Shared by `acmad.traffic` (grants) and `acmad.sim.models` (service models).
"""
from __future__ import annotations

import bisect
import math

import numpy as np

# NR MCS table 2 (TS 38.214 Table 5.1.3.1-2, 256QAM): (Qm, R x 1024)
MCS_T2: tuple[tuple[int, float], ...] = (
    (2, 120), (2, 193), (2, 308), (2, 449), (2, 602), (4, 378), (4, 434), (4, 490), (4, 553), (4, 616),
    (4, 658), (6, 466), (6, 517), (6, 567), (6, 616), (6, 666), (6, 719), (6, 772), (6, 822), (6, 873),
    (8, 682.5), (8, 711), (8, 754), (8, 797), (8, 841), (8, 885), (8, 916.5), (8, 948))
N_MCS = len(MCS_T2)
NR_EFF = np.array([q * r / 1024 for q, r in MCS_T2])          # bits per RE

TBS_SMALL = (24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192,
             208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576,
             608, 640, 672, 704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256,
             1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216,
             2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824)
LIFTING = sorted({a * 2**j for a in (2, 3, 5, 7, 9, 11, 13, 15) for j in range(8) if a * 2**j <= 384})


def tbs_bits(mcs: int, nprb: int, n_re: int = 156, layers: int = 1) -> int:
    """TBS in bits, TS 38.214 §5.1.3.2 (MCS table 2)."""
    qm, r = MCS_T2[mcs]
    r /= 1024
    n_info = min(156, n_re) * nprb * r * qm * layers
    if n_info <= 3824:
        n = max(3, math.floor(math.log2(n_info)) - 6)
        return TBS_SMALL[bisect.bisect_left(TBS_SMALL, max(24, 2**n * math.floor(n_info / 2**n)))]
    n = math.floor(math.log2(n_info - 24)) - 5
    npr = max(3840, 2**n * round((n_info - 24) / 2**n))
    c = math.ceil((npr + 24) / 3816) if r <= 0.25 else (math.ceil((npr + 24) / 8424) if npr > 8424 else 1)
    return 8 * c * math.ceil((npr + 24) / (8 * c)) - 24


def code_blocks(tbs: int, mcs: int) -> tuple[int, int, int]:
    """(base graph, number of code blocks, code-block length), TS 38.212 §5.2.2 / §7.2.2."""
    r = MCS_T2[mcs][1] / 1024
    bg = 2 if (tbs <= 292 or (tbs <= 3824 and r <= 0.67) or r <= 0.25) else 1
    b = tbs + (16 if tbs <= 3824 else 24)
    kcb = 8448 if bg == 1 else 3840
    c, bp = (1, b) if b <= kcb else (math.ceil(b / (kcb - 24)), b + math.ceil(b / (kcb - 24)) * 24)
    kb = 22 if bg == 1 else (10 if b > 640 else 9 if b > 560 else 8 if b > 192 else 6)
    z = min(z for z in LIFTING if kb * z >= bp / c)
    return bg, c, (22 if bg == 1 else 10) * z


def tbs_table(pusch_prb: int, re_per_prb: int, layers: int) -> np.ndarray:
    return np.array([[tbs_bits(m, n, re_per_prb, layers) for n in range(1, pusch_prb + 1)] for m in range(N_MCS)])


def cb_table(tbs: np.ndarray) -> np.ndarray:
    return np.array([[code_blocks(tbs[m, n], m) for n in range(tbs.shape[1])] for m in range(N_MCS)])
