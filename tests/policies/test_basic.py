import numpy as np

from acmad.interface import NULL_MESSAGE, ExecutorView, ServerObs, ServerReport, TaskView, make
from acmad.interface.observations import ClusterState, ExecutorInfo, JobView, SwitchObs
import acmad.policies  # noqa: F401


def _task():
    return TaskView(1, 0, "ldpc_decoding", np.zeros(10), 3.0, 2.0)


def _switch_obs(n_servers):
    job = JobView(0, 0, 0.0, 3.0, 3.0, (_task(),))
    reports = tuple(ServerReport(np.zeros(2), 0.0, 0.0) for _ in range(n_servers))
    cluster = ClusterState(0.0, 0.0, np.zeros(n_servers), np.zeros(n_servers))
    topo = tuple((ExecutorInfo("cpu", 1),) for _ in range(n_servers))
    return SwitchObs(0.0, job, cluster, reports, topo)


def _server_obs(server_id, allowed):
    execs = tuple(ExecutorView("gpu", 1, 0, 0, 0.0, 0, a) for a in allowed)
    return ServerObs(0.0, server_id, _task(), execs, NULL_MESSAGE)


def test_switch_round_robin_cycles_and_resets():
    p = make("switch", "round_robin")
    p.reset(0)
    obs = _switch_obs(3)
    assert [p.act(obs).server for _ in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    assert p.act(obs).message is NULL_MESSAGE
    p.reset(0)
    assert p.act(obs).server == 0


def test_server_round_robin_skips_disallowed_and_keeps_one_cursor_per_server():
    p = make("server", "round_robin")
    p.reset(0)
    a = _server_obs(0, [False, True, True])
    b = _server_obs(1, [True, True])
    assert [p.act(a).executor for _ in range(4)] == [1, 2, 1, 2]
    assert [p.act(b).executor for _ in range(3)] == [0, 1, 0]
