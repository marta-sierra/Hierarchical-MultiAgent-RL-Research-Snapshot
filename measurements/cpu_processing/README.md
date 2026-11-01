# `cpu_processing` service-time measurement (ADR-0008)

MATLAB (5G Toolbox) times the PUSCH receive blocks that make up the `cpu_processing` task, on
the configurations the simulator produces. The fitted formula and the calibration to
CloudRIC's server level are built from `times.csv` afterwards, in the repo (ADR-0008).

| File | What |
|---|---|
| `grid.csv` | the 504 configurations: every NR MCS of table 2 (0–27) × 18 nPRB values (1–250), with the TBS, base graph, code blocks, coded bits `G` and SNR our generator uses. Rebuilt by `scripts/export_cpu_processing_grid.py` |
| `measureCpuProcessing.m` | one configuration and seed: builds the grant, then times each block over the repetitions |
| `runCpuProcessing.m` | the whole run over `grid.csv`; resumable |
| `results/<host>_<date>_v<n>/` | `times.csv` + `env.txt` (written by the run; commit them when the run is complete). The run in use is `RUN` in `src/acmad/sim/models/matlab_cpu_processing.py` |

## Run
In MATLAB (R2020a or later, 5G Toolbox), from this folder:
```matlab
runCpuProcessing
```
- It runs on **one thread** (`maxNumCompThreads(1)`). Plug the laptop in, close other
  programs, and do not use the machine during the run.
- It prints progress and an ETA. It can be stopped (Ctrl-C) and started again: what is already
  in `times.csv` is skipped. Change the settings at the top of `runCpuProcessing.m` only before
  the first start.
- Every configuration asserts that MATLAB's TBS, `G` and code-block segmentation equal
  `grid.csv`. A failed assertion is a mismatch between `acmad.nr` and the 5G Toolbox: report
  it, do not skip it.

## What is measured
Numerology of the simulator: 30 kHz SCS, 273-PRB carrier, 14 symbols, 1 DMRS symbol with
2 CDM groups without data (156 RE/PRB), 1 layer, **1 TX and 1 RX antenna**. The channel is
AWGN on the resource grid (OFDM demodulation is not part of `cpu_processing`, so the time
domain is skipped).

| Column | Block | Part of `cpu_processing` |
|---|---|---|
| `t_chest_ms` | `nrChannelEstimate` (DMRS) | yes (2) |
| `t_extract_ms` | `nrExtractResources` | yes (3) |
| `t_eq_ms` | `nrEqualizeMMSE` | yes (4) |
| `t_demod_ms` | `nrPUSCHDecode`: layer demapping, soft demodulation (LLR), descrambling | yes (5) |
| `t_ratercv_ms` | `nrRateRecoverLDPC` | yes (6) |
| `t_ldpc_ms`, `ldpc_iters`, `crc_ok` | `nrLDPCDecode` (normalized min-sum, **always 10 iterations**, `Termination = 'max'`, like CloudRIC's FlexRAN CPU decoder) | **no**: only to calibrate the factor k against CloudRIC; timed in the first 5 repetitions |

Before measuring, 20 untimed runs warm MATLAB up (v1 showed the first configurations 2–4× slower).
Per (configuration, seed): 2 warm-up repetitions (discarded), then 30 kept, one row each.
5 seeds (different transport block and noise). The whole chain re-runs on the same received
grid in every repetition, each block under its own `tic` / `toc`.

## Runs
| Run | Status |
|---|---|
| `…_20261008-v1` (not committed) | superseded: LDPC with early termination (median 4 iterations vs CloudRIC's fixed 10), so k is not comparable; first configurations slowed by MATLAB warm-up. Its blocks 2–6 chose the formula (ADR-0008, amendment) |
| `…_20261008_v2` (not committed) | superseded: ran on **10 threads** (`env.txt`: `threads: 10 (was 1)`): a stale `onCleanup` from v1 restored the thread count. Blocks 2–6 ≈ 12 % slower than v1, LDPC 1.5–3.4× slower (uneven with size), so its k ≈ 50 is inflated |
| `…_20261008_v3` (**in use**, committed) | one thread, 10 LDPC iterations, global warm-up; 75 600 rows, 0 duplicates, 3 of 2 520 cells > 1.5× their configuration. k = 30.5 (ADR-0008, amendment 3) |
