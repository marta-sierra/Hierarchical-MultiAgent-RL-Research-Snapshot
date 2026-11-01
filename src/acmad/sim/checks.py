"""Invariants of the mechanics, checked on the records of a finished episode (docs/SIM_VALIDATION.md).

Only `Records.to_frame()`, the stream and the topology are read, so any episode can be checked,
however long. Each check returns the task ids that violate it (empty = holds).

Times are compared with a tolerance `TOL_MS`: event times are sums of float service times.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from acmad.sim.records import OUTCOMES
from acmad.sim.topology import Topology
from acmad.traffic.stream import TaskStream

TOL_MS = 1e-9


@dataclass(frozen=True)
class Check:
    name: str
    violations: list[int] = field(default_factory=list)     # task ids

    @property
    def ok(self) -> bool:
        return not self.violations


def check_records(records: pd.DataFrame, stream: TaskStream, topology: Topology, deadline_ms: float) -> list[Check]:
    """Every invariant on one episode: `records` = `Records.to_frame()`, D = `deadline_ms`."""
    df = records.set_index("task_id", drop=False).sort_index()
    out = [
        Check("conservation: every task of the stream ends exactly once, with a known outcome", _conservation(df, stream)),
        Check("deadline = arrival + D", _ids(df, (df.deadline_ms - df.t_arrival_ms - deadline_ms).abs() > TOL_MS)),
        Check("causality: arrival <= ready <= start <= end", _causality(df)),
        Check("nothing ends after its deadline", _ids(df, df.t_end_ms > df.deadline_ms + TOL_MS)),
        Check("outcome matches times: met ends by D, missed is aborted at D after starting, dropped never starts",
              _outcomes(df)),
        Check("dependency: ldpc starts after its cpu_processing is met, on the same server", _dependency(df, stream)),
        Check("rule: cpu_processing runs on the CPU (ADR-0011)", _rule(df, topology)),
    ]
    cap, wc, fifo = [], [], []
    for (k, e), g in df[df.executor.notna()].groupby(["server", "executor"]):
        c = topology.servers[int(k)][int(e)].capacity
        cap += _capacity(g, c)
        wc += _work_conserving(g, c)
        fifo += _fifo(g)
    out += [Check("capacity: never more tasks in service than units", cap),
            Check("work-conserving: no task waits while a unit of its executor is free", wc),
            Check("FIFO: an executor starts tasks in the order they were queued", fifo)]
    return out


def _ids(df: pd.DataFrame, mask) -> list[int]:
    return df.task_id[np.asarray(mask, dtype=bool)].astype(int).tolist()


def _conservation(df: pd.DataFrame, stream: TaskStream) -> list[int]:
    want = set(stream.tasks.task_id.astype(int))
    got = df.task_id.astype(int)
    bad = set(got[got.duplicated()]) | (want ^ set(got))
    bad |= set(_ids(df, ~df.outcome.isin(OUTCOMES)))
    return sorted(bad)


def _causality(df: pd.DataFrame) -> list[int]:
    seq = [df.t_arrival_ms, df.t_ready_ms, df.t_start_ms, df.t_end_ms]
    bad = np.zeros(len(df), dtype=bool)
    for a, b in zip(seq, seq[1:]):
        bad |= (a.to_numpy(float) > b.to_numpy(float) + TOL_MS)          # NaN compares False: skipped stages
    bad |= (df.t_arrival_ms > df.t_end_ms + TOL_MS).to_numpy()
    return _ids(df, bad)


def _outcomes(df: pd.DataFrame) -> list[int]:
    started = df.t_start_ms.notna()
    met = (df.outcome == "met") & (~started | (df.t_end_ms > df.deadline_ms + TOL_MS))
    missed = (df.outcome == "missed") & (~started | ((df.t_end_ms - df.deadline_ms).abs() > TOL_MS))
    dropped = (df.outcome == "dropped") & started
    return _ids(df, met | missed | dropped)


def _dependency(df: pd.DataFrame, stream: TaskStream) -> list[int]:
    dep = stream.tasks.loc[stream.tasks.depends_on.notna(), ["task_id", "depends_on"]].astype(int)
    dep = dep[dep.task_id.isin(df.index) & dep.depends_on.isin(df.index)]
    child, parent = df.loc[dep.task_id], df.loc[dep.depends_on]
    started = child.t_start_ms.notna().to_numpy()
    bad = started & ((parent.outcome.to_numpy() != "met")
                     | (child.t_start_ms.to_numpy(float) < parent.t_end_ms.to_numpy(float) - TOL_MS)
                     | (child.server.to_numpy() != parent.server.to_numpy()))
    return child.task_id[bad].astype(int).tolist()


def _rule(df: pd.DataFrame, topology: Topology) -> list[int]:
    cp = df[(df.kind == "cpu_processing") & df.executor.notna()]
    kinds = [topology.servers[int(k)][int(e)].kind for k, e in zip(cp.server, cp.executor)]
    return cp.task_id[np.array(kinds) != "cpu"].astype(int).tolist()


def _busy_steps(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Units in service on one executor as a step function: (change times, busy count from each time on).

    At equal times the ends are applied before the starts: a unit freed at t can serve again at t.
    """
    s = g[g.t_start_ms.notna()]
    t = np.r_[s.t_start_ms.to_numpy(float), s.t_end_ms.to_numpy(float)]
    d = np.r_[np.ones(len(s)), -np.ones(len(s))]
    order = np.lexsort((d, t))                       # by time, then -1 before +1
    t, busy = t[order], np.cumsum(d[order])
    last = np.r_[t[1:] != t[:-1], True]             # value after every change at that instant
    return t[last], busy[last]


def _capacity(g: pd.DataFrame, cap: int) -> list[int]:
    t, busy = _busy_steps(g)
    if not len(t) or busy.max() <= cap:
        return []
    over = [(a, b) for a, b, n in zip(t, np.r_[t[1:], np.inf], busy) if n > cap]
    s = g[g.t_start_ms.notna()]
    hit = np.zeros(len(s), dtype=bool)
    for a, b in over:
        hit |= (s.t_start_ms.to_numpy(float) < b) & (s.t_end_ms.to_numpy(float) > a)
    return s.task_id[hit].astype(int).tolist()


def _work_conserving(g: pd.DataFrame, cap: int) -> list[int]:
    """A task waits over [t_ready, t_start) (or until it is dropped); all `cap` units must be busy then."""
    t, busy = _busy_steps(g)
    free_from = t[busy < cap]                                   # segments with a free unit
    free_to = np.r_[t[1:], np.inf][busy < cap]
    if not len(t):
        free_from, free_to = np.array([-np.inf]), np.array([np.inf])
    else:
        free_from, free_to = np.r_[-np.inf, free_from], np.r_[t[0], free_to]   # before the first start: idle
    a = g.t_ready_ms.to_numpy(float)
    b = np.where(g.t_start_ms.notna(), g.t_start_ms.to_numpy(float), g.t_end_ms.to_numpy(float))
    waits = b - a > TOL_MS
    i = np.searchsorted(free_to, a + TOL_MS, side="right")       # first free segment ending after a
    i = np.minimum(i, len(free_to) - 1)
    bad = waits & (free_to[i] > a + TOL_MS) & (free_from[i] < b - TOL_MS)
    return g.task_id[bad].astype(int).tolist()


def _fifo(g: pd.DataFrame) -> list[int]:
    """Queued earlier (strictly smaller t_ready) => started no later. Ties in t_ready may go either way."""
    s = g[g.t_start_ms.notna()].sort_values(["t_ready_ms", "t_start_ms"])
    r, st = s.t_ready_ms.to_numpy(float), s.t_start_ms.to_numpy(float)
    first = np.r_[True, r[1:] > r[:-1] + TOL_MS]                  # first row of each t_ready group
    grp = np.cumsum(first) - 1
    gmax = np.maximum.reduceat(st, np.flatnonzero(first))         # latest start inside each group
    before = np.r_[-np.inf, np.maximum.accumulate(gmax)[:-1]]     # latest start of all earlier groups
    return s.task_id[st < before[grp] - TOL_MS].astype(int).tolist()
