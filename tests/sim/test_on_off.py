"""Executor on / off (ADR-0004): switch-on with work, switch-off T_off after going idle."""
import numpy as np
import pytest

from acmad.sim import SimConfig, Simulation

from simhelpers import FixedServer, FixedSwitch, RandomServer, RandomSwitch, make_stream, run, topology

# cpu_processing 0.5 ms on the cpu, then ldpc_decoding 1.0 ms on the gpu (executor 1)
SERVERS = [[{"kind": "cpu", "capacity": 1, "t_off_ms": 5.0}, {"kind": "gpu", "capacity": 1, "model": "test_static", "t_off_ms": 5.0}]]


def periods(arrivals, servers=SERVERS, deadline_ms=10.0):
    sim = Simulation(make_stream(arrivals), topology(servers), SimConfig(deadline_ms=deadline_ms))
    rec = run(sim, FixedSwitch(0), FixedServer(1))
    return rec.to_frame().set_index("task_id"), rec.on_periods_frame()


def test_one_job_one_period_per_executor():
    tasks, p = periods([0.0])
    gpu = p[p.executor == 1].iloc[0]
    assert (gpu.t_on_ms, gpu.t_off_ms, gpu.n_tasks) == (0.5, 6.5, 1)       # on 0.5, idle 1.5, off 1.5 + 5
    assert gpu.static_j == pytest.approx(10.0 * 6.0e-3)                     # 10 W x 6 ms
    cpu = p[p.executor == 0].iloc[0]
    assert (cpu.t_on_ms, cpu.t_off_ms, cpu.static_j) == (0.0, 5.5, 0.0)    # cpu static power 0
    assert list(tasks.on_period) == [0, 0]


@pytest.mark.parametrize("gap, n_gpu_periods", [(4.9, 1), (5.0, 2), (7.0, 2)])
def test_switch_off_after_t_off_idle(gap, n_gpu_periods):
    """GPU idle from 1.5 ms; the second ldpc reaches it at 1.5 + gap (work exactly at T_off finds it off)."""
    tasks, p = periods([0.0, gap + 1.0])
    gpu = p[p.executor == 1]
    assert len(gpu) == n_gpu_periods
    if n_gpu_periods == 2:
        assert list(gpu.t_off_ms) == [6.5, gap + 1.5 + 1.0 + 5.0]
        assert list(tasks.loc[[1, 3], "on_period"]) == [0, 1]
    else:
        assert gpu.iloc[0].t_off_ms == gap + 1.5 + 1.0 + 5.0 and gpu.iloc[0].n_tasks == 2


def test_queued_work_keeps_it_on_and_drops_have_no_period():
    # three jobs at once: the gpu serves the ldpc tasks back to back; the third is dropped in the FIFO
    tasks, p = periods([0.0, 0.0, 0.0], deadline_ms=2.6)
    gpu = p[p.executor == 1]
    assert len(gpu) == 1
    started = tasks[tasks.t_start_ms.notna() & (tasks.executor == 1)]
    assert gpu.iloc[0].n_tasks == len(started)
    assert tasks.loc[tasks.t_start_ms.isna(), "on_period"].isna().all()


def test_unused_executor_never_switches_on():
    servers = SERVERS + [[{"kind": "cpu"}, {"kind": "gpu", "model": "test_static"}]]
    _, p = periods([0.0, 1.0], servers)
    assert set(p.server) == {0}


def test_periods_are_consistent_on_a_random_run():
    stream = make_stream(list(np.sort(np.random.default_rng(0).uniform(0, 200, 300))))
    servers = [[{"kind": "cpu", "capacity": 2}, {"kind": "gpu", "model": "test_static", "t_off_ms": 3.0}]] * 2
    rec = run(Simulation(stream, topology(servers), SimConfig(deadline_ms=4.0)), RandomSwitch(2), RandomServer())
    tasks, p = rec.to_frame(), rec.on_periods_frame()
    for (k, e), g in p.groupby(["server", "executor"]):
        g = g.sort_values("on_period")
        assert list(g.on_period) == list(range(len(g)))
        assert (g.t_off_ms.to_numpy()[:-1] <= g.t_on_ms.to_numpy()[1:]).all()     # no overlap
        assert (g.t_off_ms - g.t_on_ms >= 3.0 - 1e-12).all() or g.kind.iloc[0] == "cpu"
        ran = tasks[(tasks.server == k) & (tasks.executor == e) & tasks.t_start_ms.notna()]
        assert g.n_tasks.sum() == len(ran)
        for r in ran.itertuples():                 # every task started inside its period
            per = g.set_index("on_period").loc[r.on_period]
            assert per.t_on_ms <= r.t_start_ms <= r.t_end_ms <= per.t_off_ms


def test_last_period_clipped_at_episode_end():
    """With a long T_off the last on-period ends at the episode end, not T_off after (ADR-0004)."""
    stream = make_stream([0.0])
    stream.meta["T_ms"] = 20.0
    servers = [[{"kind": "cpu", "capacity": 1}, {"kind": "gpu", "model": "test_static", "t_off_ms": 10_000.0}]]
    sim = Simulation(stream, topology(servers), SimConfig(deadline_ms=10.0))
    rec = run(sim, FixedSwitch(0), FixedServer(1))
    p = rec.on_periods_frame()
    assert (p.t_off_ms == 20.0).all()
    gpu = p[p.kind == "gpu"].iloc[0]
    assert gpu.static_j == pytest.approx(10.0 * (20.0 - gpu.t_on_ms) * 1e-3)
