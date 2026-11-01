"""`acmad.sim.checks`: the invariants hold on random and real episodes, and each one catches its violation."""
import numpy as np
import pytest

from acmad.sim import SimConfig, Simulation, check_records, load_topology

from simhelpers import RandomServer, RandomSwitch, make_stream, run, topology

SERVERS = [[{"kind": "cpu", "capacity": 2, "model": "synthetic", "params": {"dist": "exp", "mean": {"cpu_processing": 0.3, "ldpc_decoding": 1.0}}},
            {"kind": "gpu", "capacity": 1, "model": "synthetic", "params": {"dist": "exp", "mean": {"cpu_processing": 0.3, "ldpc_decoding": 0.4}}}],
           [{"kind": "cpu", "capacity": 1, "model": "synthetic", "params": {"dist": "exp", "mean": {"cpu_processing": 0.2, "ldpc_decoding": 0.8}}},
            {"kind": "gpu", "capacity": 2, "model": "synthetic", "params": {"dist": "det", "mean": {"cpu_processing": 0.3, "ldpc_decoding": 0.6}}}]]
D = 3.0


def _episode(seed=0, n=3000, rate_per_ms=2.0, downlink_ms=0.0):
    """A loaded random run: Poisson arrivals on the 0.5 ms grid's scale, random dispatch; some deadlines missed."""
    rng = np.random.default_rng(seed)
    stream = make_stream(list(np.cumsum(rng.exponential(1 / rate_per_ms, n))))
    topo = topology(SERVERS)
    sim = Simulation(stream, topo, SimConfig.from_dict({"deadline_ms": D, "links": {"downlink_ms": downlink_ms}}), seed=seed)
    df = run(sim, RandomSwitch(2), RandomServer(), seed=seed).to_frame()
    return df, stream, topo


def _failed(checks):
    return {c.name: c.violations[:5] for c in checks if not c.ok}


@pytest.mark.parametrize("seed,downlink_ms", [(0, 0.0), (1, 0.0), (2, 0.4)])
def test_invariants_hold_on_random_runs(seed, downlink_ms):
    df, stream, topo = _episode(seed, downlink_ms=downlink_ms)
    assert {"met", "missed", "dropped"} <= set(df.outcome)              # every path is exercised
    assert not _failed(check_records(df, stream, topo, D))


def _violated(df, stream, topo):
    return {c.name.split(":")[0] for c in check_records(df, stream, topo, D) if not c.ok}


def test_each_check_catches_its_violation():
    df, stream, topo = _episode(0)
    started = df[df.t_start_ms.notna() & (df.outcome == "met")]
    waited = started[started.t_start_ms > started.t_ready_ms + 1e-6]

    bad = df.drop(index=df.index[0])
    assert "conservation" in _violated(bad, stream, topo)

    bad = df.copy(); bad.loc[0, "deadline_ms"] += 0.1
    assert "deadline = arrival + D" in _violated(bad, stream, topo)

    i = started.index[0]
    bad = df.copy(); bad.loc[i, "t_start_ms"] = bad.loc[i, "t_end_ms"] + 1
    assert "causality" in _violated(bad, stream, topo)

    bad = df.copy(); bad.loc[i, "outcome"] = "dropped"
    assert "outcome matches times" in _violated(bad, stream, topo)

    ldpc = started[(started.kind == "ldpc_decoding")].index[0]
    bad = df.copy(); bad.loc[ldpc, "server"] = 1 - bad.loc[ldpc, "server"]
    assert "dependency" in _violated(bad, stream, topo)

    cpu = started[started.kind == "cpu_processing"].index[0]
    bad = df.copy(); bad.loc[cpu, "executor"] = 1
    assert "rule" in _violated(bad, stream, topo)

    # a task that waited, started as soon as it was queued: a unit too many (capacity) ...
    j = waited.index[0]
    bad = df.copy(); dt = bad.loc[j, "t_start_ms"] - bad.loc[j, "t_ready_ms"]
    bad.loc[j, ["t_start_ms", "t_end_ms"]] -= dt
    assert "capacity" in _violated(bad, stream, topo)
    # ... a task that waited, started later than it could: a unit idle (work-conserving)
    k = started[started.t_start_ms == started.t_ready_ms].index[0]
    bad = df.copy(); bad.loc[k, ["t_start_ms", "t_end_ms"]] += 50.0      # waits 50 ms with units free
    assert "work-conserving" in _violated(bad, stream, topo)


def test_fifo_catches_overtaking():
    df, stream, topo = _episode(0)
    s = df[df.t_start_ms.notna()].sort_values("t_ready_ms")
    for (k, e), g in s.groupby(["server", "executor"]):
        g = g[g.t_start_ms > g.t_ready_ms + 1e-6]           # two tasks that queued, at distinct t_ready
        if len(g) >= 2 and g.t_ready_ms.iloc[0] < g.t_ready_ms.iloc[1]:
            a, b = g.index[0], g.index[1]
            break
    bad = df.copy()
    bad.loc[[a, b], "t_start_ms"] = df.loc[[b, a], "t_start_ms"].to_numpy()   # swap the start times
    assert "FIFO" in _violated(bad, stream, topo)


@pytest.mark.trace
@pytest.mark.parametrize("policies", [("round_robin", "round_robin"), ("random", "random")])
def test_invariants_hold_on_a_full_real_episode(policies):
    """20 s of real traffic (3 DUs, s = 2) on the reference topology (ADR-0009) with the real models."""
    import acmad.policies  # noqa: F401
    from acmad.interface import make
    from acmad.traffic import generate, load_config

    cfg = load_config()
    if not cfg.trace_path.exists() or not (cfg.cloudric_dir / "CPU_dataset.csv").exists():
        pytest.skip("real trace or CloudRIC data not available")
    stream = generate(cfg, "train", 3, 2.0, 0)
    topo = load_topology("configs/topology/example.yaml")
    sim = Simulation(stream, topo, SimConfig(deadline_ms=D), seed=0)
    if policies[0] == "random":
        sw, sv = RandomSwitch(topo.n_servers), RandomServer()
    else:
        sw, sv = make("switch", policies[0]), make("server", policies[1])
    df = run(sim, sw, sv).to_frame()
    assert len(df) == len(stream.tasks) > 100_000
    assert not _failed(check_records(df, stream, topo, D))
