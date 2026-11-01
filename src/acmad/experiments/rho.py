"""Offered load rho, the "heterogeneous Erlang" of an episode (ADR-0009).

rho = the smallest possible utilisation of the most loaded executor over every fluid (fractional)
assignment of the episode's work that the routing allows: jobs go whole to a server, the work fixed
by rule (`cpu_processing`) runs on that server's CPU, the rest (`ldpc_decoding`) on any executor of
that server that can run it. Work of task i on executor e = E[sigma(e, i)] (closed-form mean of the
service model, ADR-0009 §2). rho < 1: some dispatching can serve the long-run work; rho > 1: none can.

It depends on the TaskStream and the topology only, never on a policy or the simulator state; it is
reported with every run to interpret the results (not a generator input).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from acmad.sim.models import make_model
from acmad.sim.models.base import ExecutorSpec
from acmad.sim.rules import cpu_index, rule_executor
from acmad.sim.topology import Topology
from acmad.traffic.stream import TaskStream

N_CLASSES = 100            # job classes by quantiles of the work ratio between the two executor types
PEAK_WINDOW_MS = 100.0     # rho_peak: windows of this length ...
PEAK_QUANTILE = 95.0       # ... and this percentile of their rho (ADR-0009 §5)


@dataclass(frozen=True)
class JobWork:
    """Work of every job of a stream on every executor of a topology (ms, means of the models).

    - `pinned[j, e]`: work of job j fixed by rule on executor e (the CPU of e's server), 0 elsewhere;
    - `free[j, e]`: work of job j's policy-routed tasks if they run on e (NaN: e cannot run them);
    - `server_of[e]`: server of executor e (executors flattened server by server);
    - `type_of[e]`: index of e's distinct service model (model and params): same type, same work;
    - `capacity[e]`: parallel units c_e; `arrival_ms[j]`; `T_ms`: episode length.
    """

    pinned: np.ndarray
    free: np.ndarray
    server_of: np.ndarray
    type_of: np.ndarray
    capacity: np.ndarray
    arrival_ms: np.ndarray
    T_ms: float


@dataclass(frozen=True)
class Rho:
    """rho with the optimal fluid assignment behind it.

    `utilisation[e]`: load / capacity of executor e at the optimum (its max is `rho`);
    `y[c, k]`: share of job class c sent to server k; `z[c, e]`: share of class c whose free work
    runs on executor e; `job_class[j]`: class of job j.
    """

    rho: float
    utilisation: np.ndarray
    y: np.ndarray
    z: np.ndarray
    job_class: np.ndarray


def _spec_key(spec: ExecutorSpec) -> str:
    return json.dumps([spec.model_name, spec.params], sort_keys=True, default=str)


def job_work(stream: TaskStream, topology: Topology, T_ms: float | None = None) -> JobWork:
    """Mean work of every job on every executor (ADR-0009 §2)."""
    tasks = stream.tasks.sort_values("task_id", kind="stable")
    jobs = stream.jobs.reset_index(drop=True)
    row = pd.Series(np.arange(len(jobs)), index=jobs.job_id.to_numpy())
    jrow = row.loc[tasks.job_id.to_numpy()].to_numpy()
    allowed = tasks.allowed.astype(str).to_numpy()
    if not np.isin(allowed, ("any", "cpu")).all():
        raise NotImplementedError(f"allowed values {set(allowed) - {'any', 'cpu'}} not handled")
    pinned_task = np.array([rule_executor(k, a, 0) is not None
                            for k, a in zip(tasks.kind.astype(str).to_numpy(), allowed)])

    specs = [s for srv in topology.servers for s in srv]
    server_of = np.array([k for k, srv in enumerate(topology.servers) for _ in srv])
    cpu_of = {k: cpu_index(list(srv)) for k, srv in enumerate(topology.servers)}
    first = {k: int(np.flatnonzero(server_of == k)[0]) for k in cpu_of}
    means: dict[str, np.ndarray] = {}
    keys = [_spec_key(s) for s in specs]
    J, E = len(jobs), len(specs)
    pinned, free = np.zeros((J, E)), np.zeros((J, E))
    for e, spec in enumerate(specs):
        key = keys[e]
        if key not in means:
            means[key] = make_model(spec.model_name).mean_service_ms(tasks, spec)
        w = means[key]
        k = server_of[e]
        if e - first[k] == cpu_of[k]:
            if np.isnan(w[pinned_task]).any():
                raise ValueError(f"executor {e} (cpu of server {k}) cannot run a rule-routed task")
            pinned[:, e] = np.bincount(jrow[pinned_task], w[pinned_task], minlength=J)
        f = ~pinned_task
        free[:, e] = np.bincount(jrow[f], np.nan_to_num(w[f], nan=0.0), minlength=J)
        cannot = np.bincount(jrow[f], np.isnan(w[f]).astype(float), minlength=J) > 0
        free[cannot, e] = np.nan
    T = float(T_ms if T_ms is not None else stream.meta["T_ms"])
    type_of = np.array([list(dict.fromkeys(keys)).index(k) for k in keys])
    return JobWork(pinned, free, server_of, type_of, np.array([s.capacity for s in specs], dtype=float),
                   jobs.t_arrival_ms.to_numpy(float), T)


def job_classes(work: JobWork, n_classes: int | None = N_CLASSES) -> np.ndarray:
    """Class of every job (ADR-0009 §3): quantiles of the free-work ratio between the two distinct
    executor types (CPU and GPU: the fluid optimum fills the GPU in order of that ratio).
    `n_classes=None`: one class per job (exact LP)."""
    J = len(work.arrival_ms)
    if n_classes is None or J <= n_classes:
        return np.arange(J)
    types = [int(np.flatnonzero(work.type_of == t)[0]) for t in np.unique(work.type_of)]
    if len(types) == 1:
        return np.zeros(J, dtype=int)
    if len(types) > 2:
        raise NotImplementedError("job classes by ratio need at most two executor types; use n_classes=None")
    a, b = work.free[:, types[0]], work.free[:, types[1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(a > 0, b / a, 0.0)
    r = np.nan_to_num(r, nan=np.inf)
    edges = np.quantile(r, np.linspace(0, 1, n_classes + 1)[1:-1])
    return np.searchsorted(edges, r, side="right")


def _class_sums(x: np.ndarray, cls: np.ndarray, n: int) -> np.ndarray:
    """Sum the rows of `x` (J, E) by class; a NaN in any job of a class makes the class NaN."""
    out = np.zeros((n, x.shape[1]))
    np.add.at(out, cls, x)
    return out


def rho(work: JobWork, n_classes: int | None = N_CLASSES) -> Rho:
    """Solve the LP of ADR-0009 §3 (HiGHS):

        min rho  s.t.  sum_k y[c,k] = 1;  sum_{e in E_k} z[c,e] = y[c,k];
                       sum_c P[c,e] y[c,k(e)] + F[c,e] z[c,e] <= rho * c_e   for every executor e,
    with P, F the classes' pinned and free work rates (ms per ms) and z[c,e] = 0 where e cannot run c.
    """
    cls = job_classes(work, n_classes)
    C = int(cls.max()) + 1
    P = _class_sums(work.pinned, cls, C) / work.T_ms
    F = _class_sums(work.free, cls, C) / work.T_ms
    E = F.shape[1]
    K = int(work.server_of.max()) + 1
    ny, nz = C * K, C * E
    iy = lambda c, k: c * K + k                          # noqa: E731
    iz = lambda c, e: ny + c * E + e                     # noqa: E731
    irho = ny + nz

    eq_r, eq_c, eq_v, b_eq = [], [], [], []
    for c in range(C):                                   # sum_k y = 1
        eq_r += [len(b_eq)] * K; eq_c += [iy(c, k) for k in range(K)]; eq_v += [1.0] * K; b_eq.append(1.0)
    for c in range(C):                                   # sum_{e in k} z = y
        for k in range(K):
            es = np.flatnonzero(work.server_of == k)
            r = len(b_eq)
            eq_r += [r] * (len(es) + 1); eq_c += [iz(c, e) for e in es] + [iy(c, k)]
            eq_v += [1.0] * len(es) + [-1.0]; b_eq.append(0.0)
    ub_r, ub_c, ub_v = [], [], []
    for e in range(E):                                   # load_e - rho c_e <= 0
        k = work.server_of[e]
        for c in range(C):
            if P[c, e] > 0:
                ub_r.append(e); ub_c.append(iy(c, k)); ub_v.append(P[c, e])
            if np.isfinite(F[c, e]) and F[c, e] > 0:
                ub_r.append(e); ub_c.append(iz(c, e)); ub_v.append(F[c, e])
        ub_r.append(e); ub_c.append(irho); ub_v.append(-work.capacity[e])
    n = irho + 1
    bounds = [(0, None)] * ny + [(0, 0 if not np.isfinite(F[c, e]) else None) for c in range(C) for e in range(E)] \
        + [(0, None)]
    cost = np.zeros(n)
    cost[irho] = 1.0
    res = linprog(cost, A_ub=sparse.csr_matrix((ub_v, (ub_r, ub_c)), shape=(E, n)), b_ub=np.zeros(E),
                  A_eq=sparse.csr_matrix((eq_v, (eq_r, eq_c)), shape=(len(b_eq), n)), b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"rho LP failed: {res.message}")
    y = res.x[:ny].reshape(C, K)
    z = res.x[ny:irho].reshape(C, E)
    util = _load(P, np.nan_to_num(F), y, z, work.server_of) / work.capacity
    return Rho(float(res.x[irho]), util, y, z, cls)


def _load(P: np.ndarray, F: np.ndarray, y: np.ndarray, z: np.ndarray, server_of: np.ndarray) -> np.ndarray:
    """Work rate on every executor for class rates P, F under the assignment (y, z); stacks over a
    leading window axis if P, F have one."""
    return (P * y[:, server_of]).sum(-2) + (F * z).sum(-2)


def rho_peak(work: JobWork, opt: Rho, window_ms: float = PEAK_WINDOW_MS, q: float = PEAK_QUANTILE) -> float:
    """ADR-0009 §5: percentile `q` over windows of `window_ms` of the max executor utilisation, the
    dispatching held at the episode's optimum `opt` (bursts at fixed assignment)."""
    n_w = max(int(np.ceil(work.T_ms / window_ms)), 1)
    w = np.minimum((work.arrival_ms // window_ms).astype(int), n_w - 1)
    C = opt.y.shape[0]
    idx = w * C + opt.job_class
    P = np.zeros((n_w * C, work.pinned.shape[1]))
    F = np.zeros_like(P)
    np.add.at(P, idx, work.pinned)
    np.add.at(F, idx, np.nan_to_num(work.free))
    shape = (n_w, C, -1)
    load = _load(P.reshape(shape) / window_ms, F.reshape(shape) / window_ms, opt.y, opt.z, work.server_of)
    return float(np.percentile((load / work.capacity).max(axis=1), q))


def offered_load(stream: TaskStream, topology: Topology, n_classes: int | None = N_CLASSES) -> dict:
    """What every run records: rho, rho_peak and the utilisation of each executor at the optimum."""
    work = job_work(stream, topology)
    opt = rho(work, n_classes)
    return {"rho": opt.rho, "rho_peak": rho_peak(work, opt),
            "utilisation_at_optimum": opt.utilisation.round(6).tolist(), "n_classes": int(opt.y.shape[0])}
