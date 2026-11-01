import numpy as np
import pandas as pd
import pytest

from acmad.interface import Channel, ServerObs, SwitchObs
from acmad.sim import SimConfig, Simulation, load_topology
from acmad.sim.load import ArrivalRate, MeanTBS
from acmad.sim.models import ExecutorSpec, ServiceModel, TaskFeatures, make_model

from simhelpers import (
    FixedServer, FixedSwitch, RandomServer, RandomSwitch, make_stream, run, sim_for, topology,
)

TWO = [[{"kind": "cpu", "capacity": 2}, {"kind": "gpu", "capacity": 1}],
       [{"kind": "cpu", "capacity": 1}, {"kind": "gpu", "capacity": 2}]]


def _random_run(seed, n=40):
    rng = np.random.default_rng(123)
    arr = np.sort(rng.integers(0, 40, n) * 0.5)
    stream = make_stream(list(arr), list(rng.integers(1, 250, n)))
    servers = [[{"kind": "cpu", "capacity": 2, "params": {"per_prb": 0.004}},
                {"kind": "gpu", "capacity": 1, "params": {"per_prb": 0.003, "power_w": 3.0}}],
               [{"kind": "cpu", "capacity": 1, "params": {"per_prb": 0.002}},
                {"kind": "gpu", "capacity": 2, "params": {"per_prb": 0.005}}]]
    sim = Simulation(stream, topology(servers), SimConfig(), channel=Channel(3, 4))
    return run(sim, RandomSwitch(2, dim=3), RandomServer(), seed=seed).to_frame(), stream


def test_determinism():
    a, _ = _random_run(7)
    b, _ = _random_run(7)
    pd.testing.assert_frame_equal(a, b)
    c, _ = _random_run(8)
    assert not a.equals(c)


def test_conservation():
    df, stream = _random_run(3)
    assert sorted(df.task_id) == sorted(stream.tasks.task_id)
    assert set(df.outcome) <= {"met", "missed", "dropped"}
    assert (df.outcome == "met").any() and (df.outcome != "met").any()
    met = df[df.outcome == "met"]
    assert (met.t_end_ms <= met.deadline_ms).all()
    assert (df.t_end_ms <= df.deadline_ms + 1e-12).all()   # nothing outlives its deadline
    # a met ldpc task implies its cpu task was met, on the same server
    for j, g in df.groupby("job_id"):
        g = g.set_index("kind")
        if g.loc["ldpc_decoding", "outcome"] == "met":
            assert g.loc["cpu_processing", "outcome"] == "met"
        assert g.server.nunique(dropna=False) == 1


def test_messages_are_quantised_on_the_wire():
    df, _ = _random_run(1)
    levels = np.arange(16) / 15
    for m in df.message:
        assert len(m) == 3 and all(np.isclose(levels, x).any() for x in m)


def test_event_driven_and_rule_routing():
    stream = make_stream([0.0, 0.0, 1.0, 5.0])
    sim = sim_for(stream, TWO)
    log = []
    df = run(sim, RandomSwitch(2), RandomServer(), log=log).to_frame()
    sw = [r for r in log if r.tier == "switch"]
    sv = [r for r in log if r.tier == "server"]
    assert [r.obs.job.id for r in sw] == [0, 1, 2, 3]          # ties by job_id
    assert all(r.obs.now_ms == r.obs.job.arrival_ms for r in sw)
    assert all(r.obs.task.kind == "ldpc_decoding" for r in sv)  # cpu tasks never ask a policy
    cpu = df[df.kind == "cpu_processing"].set_index("job_id")
    ldpc = df[df.kind == "ldpc_decoding"].set_index("job_id")
    assert cpu.server_decision_id.isna().all() and (cpu.executor == 0).all()
    for r in sv:   # each server request happens exactly when its task entered the ready queue
        j = r.obs.task.job_id
        assert r.obs.now_ms == ldpc.loc[j, "t_ready_ms"] == cpu.loc[j, "t_end_ms"]
    assert len(sv) == ldpc.server_decision_id.notna().sum()


def test_information_structure():
    stream = make_stream([0.0, 0.5])
    log = []
    run(sim_for(stream, TWO), RandomSwitch(2), RandomServer(), log=log)
    for r in log:
        if r.tier == "server":
            assert isinstance(r.obs, ServerObs)
            assert r.obs.oracle_cluster is None and r.obs.oracle_service is None
            assert len(r.obs.executors) == 2      # its own executors only
        else:
            assert isinstance(r.obs, SwitchObs)
            assert set(SwitchObs.__dataclass_fields__) == {"now_ms", "job", "cluster", "reports", "topology",
                                                           "estimates"}
            assert len(r.obs.reports) == 2
            assert [[(e.kind, e.capacity) for e in s] for s in r.obs.topology] == \
                   [[("cpu", 2), ("gpu", 1)], [("cpu", 1), ("gpu", 2)]]
    log = []
    run(sim_for(stream, TWO, oracle_cluster=True), RandomSwitch(2), RandomServer(), log=log)
    assert all(r.obs.oracle_cluster is not None for r in log if r.tier == "server")


def test_service_oracle_is_exact():
    """ADR-0014: the service oracle shows the very (sigma, energy) the task then gets."""
    rng = np.random.default_rng(5)
    arr = np.sort(rng.integers(0, 20, 15) * 0.5)
    stream = make_stream(list(arr), list(rng.integers(1, 250, 15)))
    topo = load_topology("configs/topology/example.yaml")
    log = []
    sim = Simulation(stream, topo, SimConfig(oracle_service=True, deadline_ms=100.0), seed=3)
    df = run(sim, RandomSwitch(3), RandomServer(), log=log).to_frame().set_index("task_id")
    reqs = [r for r in log if r.tier == "server"]
    assert reqs
    for r in reqs:
        o = r.obs.oracle_service
        assert all((v is None) == (not e.allowed) for v, e in zip(o, r.obs.executors))
        row = df.loc[r.obs.task.id]
        sigma, energy = o[int(row.executor)]
        assert row.outcome == "met"
        assert np.isclose(row.t_end_ms - row.t_start_ms, sigma) and np.isclose(row.energy_dynamic_j, energy)


def test_cluster_state_counts_dispatches_not_yet_acked():
    """c_t (ADR-0014): jobs/bits sent to k and delivered after k's newest report; offered load."""
    stream = make_stream([0.0, 0.2, 1.0])         # cpu task 0.5 ms, ldpc 1.0 ms (FixedModel)
    log = []
    run(Simulation(stream, topology(TWO), SimConfig.from_dict({"links": {"downlink_ms": 0.1}})),
        FixedSwitch(0), FixedServer(1), log=log)
    sw = [r.obs.cluster for r in log if r.tier == "switch"]
    bits = int(stream.tasks.total_bits.iloc[0] + stream.tasks.total_bits.iloc[1])
    assert list(sw[0].jobs_since_ack) == [0, 0]
    assert list(sw[1].jobs_since_ack) == [1, 0] and list(sw[1].bits_since_ack) == [bits, 0]
    # t=1.0: newest report from server 0 is the ACK at 0.6 (job 0's cpu ends at 0.1 + 0.5); it
    # reflects job 0 (delivered 0.1) and job 1 (delivered 0.3)
    assert list(sw[2].jobs_since_ack) == [0, 0]
    # jobs in the last 10 ms (default window), the arriving one included
    assert [c.arrival_rate_per_ms for c in sw] == pytest.approx([0.1, 0.2, 0.3])
    assert sw[0].mean_tbs == stream.tasks.tbs.iloc[0]
    assert sw[0].vector().shape == (2 + 2 * 2,)


def test_server_sees_local_load():
    stream = make_stream([0.0, 0.5, 1.0])
    log = []
    run(sim_for(stream, TWO, deadline_ms=10.0), FixedSwitch(1), FixedServer(1), log=log)
    sv = [r.obs for r in log if r.tier == "server"]
    assert sv[-1].arrival_rate_per_ms == pytest.approx(3 / 10.0)     # 3 jobs delivered within the 10 ms window


def test_reports_age_and_ack():
    # job 0 on server 0 completes cpu at 0.5 -> ACK; switch at t=1.0 sees report age 0.5
    stream = make_stream([0.0, 1.0])
    log = []
    run(sim_for(stream, TWO), FixedSwitch(0), FixedServer(1), log=log)
    sw = [r.obs for r in log if r.tier == "switch"]
    assert sw[0].reports[0].reported_at_ms == 0.0 and sw[0].reports[0].age_ms == 0.0
    rep = sw[1].reports[0]
    assert rep.summary.shape == (2, 4)            # (n_busy, n_queued, elapsed_ms, queued_bits), ADR-0001
    assert rep.reported_at_ms == 0.5 and rep.age_ms == 0.5
    assert sw[1].estimates is not None and np.array_equal(sw[1].estimates[0], rep.summary)


def test_uplink_delay_ages_reports():
    stream = make_stream([0.0, 1.0])
    log = []
    run(Simulation(stream, topology(TWO), SimConfig.from_dict({"links": {"uplink_ms": 0.6}})),
        FixedSwitch(0), FixedServer(1), log=log)
    sw = [r.obs for r in log if r.tier == "switch"]
    assert sw[1].reports[0].reported_at_ms == 0.0      # ACK from t=0.5 still in flight


def test_capacity_k_fifo():
    # 3 jobs at t=0 on cpu capacity 2: two cpu tasks start at 0, the third at 0.5
    stream = make_stream([0.0, 0.0, 0.0])
    servers = [[{"kind": "cpu", "capacity": 2}, {"kind": "gpu", "capacity": 1}]]
    df = run(sim_for(stream, servers, deadline_ms=10.0), FixedSwitch(0), FixedServer(1)).to_frame()
    cpu = df[df.kind == "cpu_processing"].set_index("job_id")
    assert list(cpu.t_start_ms) == [0.0, 0.0, 0.5]
    gpu = df[df.kind == "ldpc_decoding"].set_index("job_id")
    # gpu capacity 1, FIFO in ready order: jobs 0,1 ready at 0.5, job 2 at 1.0
    assert list(gpu.t_start_ms) == [0.5, 1.5, 2.5]
    assert (df.outcome == "met").all()
    assert np.isclose(df.energy_dynamic_j.sum(), 3 * 0.5 + 3 * 1.0)


def test_blocked_set_releases_out_of_arrival_order():
    # job 0 has a long cpu task (big nprb), job 1 a short one; job 1's ldpc is released first
    stream = make_stream([0.0, 0.1], nprb=[200, 1])
    servers = [[{"kind": "cpu", "capacity": 2, "params": {"per_prb": 0.01}},
                {"kind": "gpu", "capacity": 1}]]
    log = []
    df = run(sim_for(stream, servers, deadline_ms=10.0), FixedSwitch(0), FixedServer(1),
             log=log).to_frame().set_index("task_id")
    order = [r.obs.task.id for r in log if r.tier == "server"]
    assert order == [3, 1]
    assert df.loc[3, "t_ready_ms"] < df.loc[1, "t_ready_ms"]
    assert df.loc[3, "t_start_ms"] < df.loc[1, "t_start_ms"]


def test_deadline_abort_with_proportional_energy():
    # cpu 0.5 ms, then ldpc on gpu 4 ms from t=0.5; deadline 3.0 -> abort at 3.0, 2.5/4 done
    stream = make_stream([0.0, 0.0])
    servers = [[{"kind": "cpu", "capacity": 2},
                {"kind": "gpu", "capacity": 1,
                 "params": {"svc": {"cpu_processing": 1.0, "ldpc_decoding": 4.0}, "power_w": 2.0}}]]
    df = run(sim_for(stream, servers, deadline_ms=3.0), FixedSwitch(0), FixedServer(1)).to_frame().set_index("task_id")
    assert df.loc[1, "outcome"] == "missed"
    assert df.loc[1, "t_end_ms"] == 3.0
    assert np.isclose(df.loc[1, "energy_dynamic_j"], 8.0 * 2.5 / 4.0)
    assert df.loc[3, "outcome"] == "dropped" and df.loc[3, "energy_dynamic_j"] == 0.0   # waited in FIFO
    assert df.loc[3, "t_start_ms"] is None or pd.isna(df.loc[3, "t_start_ms"])
    assert (df.loc[[0, 2], "outcome"] == "met").all()


def test_aborted_unit_is_freed_at_once():
    # job 0's ldpc is aborted at 3.0; job 1's ldpc (deadline 3.5) starts on the freed unit at once
    stream = make_stream([0.0, 0.5])
    servers = [[{"kind": "cpu", "capacity": 2},
                {"kind": "gpu", "capacity": 1,
                 "params": {"svc": {"cpu_processing": 1.0, "ldpc_decoding": 2.9}}}]]
    df = run(sim_for(stream, servers, deadline_ms=3.0), FixedSwitch(0), FixedServer(1)).to_frame().set_index("task_id")
    assert df.loc[1, "outcome"] == "missed" and df.loc[1, "t_end_ms"] == 3.0
    assert df.loc[3, "t_start_ms"] == 3.0          # started as soon as the unit was freed
    assert df.loc[3, "outcome"] == "missed"


def test_completion_exactly_at_deadline_is_met():
    stream = make_stream([0.0])
    servers = [[{"kind": "cpu", "capacity": 1},
                {"kind": "gpu", "capacity": 1, "params": {"svc": {"cpu_processing": 1.0, "ldpc_decoding": 2.5}}}]]
    df = run(sim_for(stream, servers, deadline_ms=3.0), FixedSwitch(0), FixedServer(1)).to_frame()
    assert (df.outcome == "met").all() and df.t_end_ms.max() == 3.0


def test_invalid_actions_rejected():
    sim = sim_for(make_stream([0.0]), TWO)
    req = sim.next_decision()
    from acmad.interface import ServerAction, SwitchAction
    with pytest.raises(TypeError):
        sim.apply(ServerAction(0))
    with pytest.raises(ValueError):
        sim.apply(SwitchAction(5))


def test_topology_validation_and_shipped_models():
    t = load_topology("configs/topology/example.yaml")
    assert t.n_servers == 3 and [len(s) for s in t.servers] == [2, 3, 1]       # ADR-0009 reference topology (GPU1, GPU2 on server 1)
    with pytest.raises(ValueError):
        topology([[{"kind": "gpu"}]])
    with pytest.raises(ValueError):
        topology([[{"kind": "cpu"}, {"kind": "cpu"}]])
    f = TaskFeatures("cpu_processing", 10, 5, 3, 1000, 2, 1, 1040, 1040, 4, 378 / 1024)
    with pytest.raises(NotImplementedError, match="not modelled yet"):          # cpu_processing is CPU-only (ADR-0011)
        make_model("gpu")(f, ExecutorSpec("gpu"), np.random.default_rng(0))


def test_sim_never_imports_policies():
    import pathlib
    import acmad.sim
    root = pathlib.Path(acmad.sim.__file__).parent
    for p in root.rglob("*.py"):
        import re
        assert not re.search(r"^\s*(from|import)\s+acmad\.policies", p.read_text(), re.M), p


def test_job_expiring_in_flight_is_dropped():
    stream = make_stream([0.0])
    sim = Simulation(stream, topology(TWO), SimConfig.from_dict({"deadline_ms": 3.0, "links": {"downlink_ms": 4.0}}))
    log = []
    df = run(sim, FixedSwitch(0), FixedServer(1), log=log).to_frame()
    assert [r.tier for r in log] == ["switch"]
    assert (df.outcome == "dropped").all() and (df.t_end_ms == 3.0).all()


def test_service_noise_is_keyed_not_ordered():
    """ADR-0002: a task's draw on an executor depends on (seed, server, executor, task) only."""
    from acmad.sim.executor import Executor

    class Noisy(ServiceModel):
        def __call__(self, task, spec, rng):
            s = 1.0 + rng.random()
            return s, 2.0 * s

    f = TaskFeatures("ldpc_decoding", 10, 5, 3, 1000, 2, 1, 1040, 1040, 4, 378 / 1024)
    def costs(order, seed=7, server=1, index=0):
        ex = Executor(ExecutorSpec("gpu"), Noisy(), seed, server, index)
        for t in order:
            ex.enqueue(t, f, 0.0)
        return {t: ex._cost[t] for t in order}
    a, b = costs([1, 2, 3]), costs([3, 1, 2])
    assert a == b
    assert costs([1], seed=8)[1] != a[1] and costs([1], index=1)[1] != a[1] and a[1] != a[2]


def test_arrival_rate_window():
    """Jobs in (now - W, now] / W; older arrivals leave the window (ADR-0014)."""
    r = ArrivalRate(2.0)
    for t in (0.0, 0.5, 1.0):
        r.add(t)
    assert r.rate_per_ms(1.0) == pytest.approx(3 / 2.0)
    assert r.rate_per_ms(2.0) == pytest.approx(2 / 2.0)     # the job at 0.0 is out: (0, 2]
    assert r.rate_per_ms(10.0) == 0.0


def test_mean_tbs_ewma():
    m = MeanTBS()
    assert m.value == 0.0
    m.update(1000.0)
    m.update(2000.0)
    assert m.value == pytest.approx(1000.0 + 0.1 * 1000.0)
