"""The mechanics against queueing theory (docs/SIM_VALIDATION.md).

One server, Poisson job arrivals, the `synthetic` service model. One executor is the queue under
test (c units, exponential or deterministic service); the other stage of the chain is a pure
delay (deterministic, many units, never queues), so the queue sees Poisson arrivals and is an
M/M/c or M/D/1 queue whose waiting time has a closed form. Deadlines are far away (no drops).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from acmad.interface import ServerAction, ServerPolicy, SwitchAction, SwitchPolicy
from acmad.sim import SimConfig, Simulation, Topology
from acmad.traffic.stream import TaskStream

NO_DEADLINE_MS = 1e9
DELAY_MS = 0.01          # service time of the non-tested stage
DELAY_UNITS = 64         # its units: never all busy at the rates used here
WARMUP = 0.1             # fraction of the jobs (by arrival) left out of the estimates
N_BATCHES = 20           # batch means for the standard error


# ---------------------------------------------------------------------- theory
def erlang_c(c: int, a: float) -> float:
    """Probability that an arrival waits in M/M/c with offered load a = λ/μ (Erlang C)."""
    rho = a / c
    if not 0 <= rho < 1:
        raise ValueError(f"unstable: rho = {rho}")
    top = a ** c / math.factorial(c) / (1 - rho)
    return top / (sum(a ** k / math.factorial(k) for k in range(c)) + top)


def theory(dist: str, c: int, mean_ms: float, rho: float) -> dict[str, float]:
    """Mean wait in queue `wq_ms` and P(wait > 0) for M/M/c (`exp`) or M/D/1 (`det`, c = 1)."""
    lam = rho * c / mean_ms
    if dist == "exp":
        p = erlang_c(c, lam * mean_ms)
        return {"wq_ms": p / (c / mean_ms - lam), "p_wait": p}
    if dist == "det" and c == 1:                       # Pollaczek-Khinchine, E[S^2] = S^2
        return {"wq_ms": lam * mean_ms ** 2 / (2 * (1 - rho)), "p_wait": rho}
    raise ValueError("closed form only for M/M/c and M/D/1")


# ---------------------------------------------------------------------- synthetic traffic
def synthetic_stream(arrivals_ms, nprb=None) -> TaskStream:
    """One job per arrival: task 2j = cpu_processing, 2j+1 = ldpc_decoding depending on it.

    Features are fixed placeholders (the synthetic model ignores them)."""
    n = len(arrivals_ms)
    nprb = np.full(n, 10) if nprb is None else np.asarray(nprb)
    ids = np.arange(n, dtype="int64")
    jobs = pd.DataFrame({"job_id": ids, "du": np.zeros(n, dtype="int16"),
                         "t_arrival_ms": np.asarray(arrivals_ms, dtype="float64")})
    depends_on = pd.array(np.repeat(2 * ids, 2), dtype="Int64")
    depends_on[::2] = pd.NA
    m = 2 * n
    tasks = pd.DataFrame({
        "task_id": np.arange(m, dtype="int64"), "job_id": np.repeat(ids, 2),
        "kind": pd.Categorical(np.tile(["cpu_processing", "ldpc_decoding"], n), categories=["cpu_processing", "ldpc_decoding"]),
        "allowed": pd.Categorical(np.tile(["cpu", "any"], n), categories=["any", "cpu"]),
        "depends_on": depends_on, "nprb": np.repeat(nprb, 2).astype("int16"),
        "mcs": np.full(m, 10, dtype="int8"), "snr_db": np.full(m, 12, dtype="int8"),
        "tbs": np.full(m, 5000, dtype="int32"), "base_graph": np.full(m, 1, dtype="int8"),
        "n_cb": np.full(m, 1, dtype="int16"), "cb_len": np.full(m, 5280, dtype="int32"),
        "total_bits": np.full(m, 5280, dtype="int32"), "qm": np.full(m, 4, dtype="int8"),
        "code_rate": np.full(m, 658 / 1024, dtype="float32"),
    })
    s = TaskStream(jobs, tasks, {})
    s.validate()
    return s


# ---------------------------------------------------------------------- the experiment
class _ToServer0(SwitchPolicy):
    def act(self, obs):
        return SwitchAction(0)


class _ToExecutor(ServerPolicy):
    def __init__(self, executor: int):
        self.executor = executor

    def act(self, obs):
        return ServerAction(self.executor)


@dataclass(frozen=True)
class QueueCase:
    """The queue under test: on the `cpu` (cpu_processing) or the `gpu` (ldpc_decoding) executor."""

    queue: str          # "cpu" | "gpu"
    dist: str           # "exp" | "det"
    c: int              # units
    mean_ms: float      # mean service time
    rho: float          # utilisation λ·mean / c

    @property
    def label(self) -> str:
        return f"M/{'M' if self.dist == 'exp' else 'D'}/{self.c} on {self.queue}, ρ={self.rho}"


def topology_for(case: QueueCase) -> Topology:
    tested = {"kind": case.queue, "capacity": case.c, "model": "synthetic",
              "params": {"dist": case.dist, "mean": {"cpu_processing": case.mean_ms, "ldpc_decoding": case.mean_ms}}}
    delay = {"kind": "gpu" if case.queue == "cpu" else "cpu", "capacity": DELAY_UNITS, "model": "synthetic",
             "params": {"dist": "det", "mean": {"cpu_processing": DELAY_MS, "ldpc_decoding": DELAY_MS}}}
    cpu, gpu = (tested, delay) if case.queue == "cpu" else (delay, tested)
    return Topology.from_dict({"servers": [{"executors": [cpu, gpu]}]})


def run_queue(case: QueueCase, n_jobs: int, seed: int = 0, keep_waits: bool = False) -> dict:
    """Simulate `n_jobs` Poisson jobs and measure the tested queue; with the theory alongside.

    `keep_waits`: also return every measured wait in queue (`waits_ms`, after the warm-up)."""
    rng = np.random.default_rng(seed)
    lam = case.rho * case.c / case.mean_ms
    stream = synthetic_stream(np.cumsum(rng.exponential(1 / lam, n_jobs)))
    sim = Simulation(stream, topology_for(case), SimConfig(deadline_ms=NO_DEADLINE_MS), seed=seed)
    sw, sv = _ToServer0(), _ToExecutor(1)              # ldpc always to executor 1 (the gpu)
    sw.reset(seed)
    sv.reset(seed + 1)
    while (req := sim.next_decision()) is not None:
        sim.apply((sw if req.tier == "switch" else sv).act(req.obs))
    df = sim.finalize().to_frame()
    q = df[df.kind == ("cpu_processing" if case.queue == "cpu" else "ldpc_decoding")].sort_values("t_ready_ms")
    if (q.outcome != "met").any():
        raise RuntimeError("a task missed its (far) deadline")
    q = q.iloc[int(WARMUP * len(q)):]
    wait = (q.t_start_ms - q.t_ready_ms).to_numpy()
    batches = np.array_split(wait, N_BATCHES)
    span = q.t_end_ms.max() - q.t_ready_ms.min()
    return {"label": case.label, **{f"case_{k}": v for k, v in case.__dict__.items()}, "n_jobs": n_jobs,
            "wq_ms": wait.mean(), "wq_se_ms": np.std([b.mean() for b in batches], ddof=1) / np.sqrt(N_BATCHES),
            "p_wait": (wait > 1e-12).mean(), "util": (q.t_end_ms - q.t_start_ms).sum() / (case.c * span),
            **{f"theory_{k}": v for k, v in theory(case.dist, case.c, case.mean_ms, case.rho).items()},
            **({"waits_ms": wait} if keep_waits else {})}
