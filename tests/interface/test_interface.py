import numpy as np
import pytest

from acmad.interface import (
    NULL_MESSAGE, Channel, ExecutorView, Message, ServerAction, ServerObs, ServerPolicy,
    ServerReport, SwitchPolicy, TaskView, available, make, make_estimator, register,
)


def _task():
    return TaskView(1, 0, "ldpc_decoding", np.array([10, 5, 3, 1000]), 3.0, 2.0)


def test_observation_arrays_are_read_only_copies():
    src = np.array([1.0, 2.0])
    r = ServerReport(src, 0.0, 1.0)
    src[0] = 9
    assert r.summary[0] == 1.0
    with pytest.raises(ValueError):
        r.summary[0] = 5
    with pytest.raises(ValueError):
        _task().features[0] = 0


def test_server_obs_has_no_cluster_field_except_oracle():
    names = set(ServerObs.__dataclass_fields__)
    assert names == {"now_ms", "server_id", "task", "executors", "message", "arrival_rate_per_ms",
                     "oracle_cluster", "oracle_service"}
    ev = ExecutorView("cpu", 1, 0, 0, 0.0, 0)
    obs = ServerObs(0.0, 0, _task(), (ev,), NULL_MESSAGE)
    assert obs.oracle_cluster is None and obs.oracle_service is None


def test_executor_view_shows_no_service_time_or_energy():
    """ADR-0001: a policy never sees sigma(e, i) or e(a), nor sums of them."""
    names = set(ExecutorView.__dataclass_fields__)
    assert names == {"kind", "capacity", "n_busy", "n_queued", "elapsed_ms", "queued_bits", "allowed"}


def test_channel_dims_and_bits():
    ch = Channel(dim=3, bits=2)
    out = ch.transmit(Message(np.array([0.1, 0.5, 2.0])))
    assert out.vector.shape == (3,)
    levels = np.linspace(0, 1, 4)
    assert all(np.isclose(levels, x).any() for x in out.vector)
    assert out.vector[2] == 1.0
    assert ch.header_bits == 6
    with pytest.raises(ValueError):
        ch.transmit(Message(np.zeros(2)))
    with pytest.raises(ValueError):
        ch.transmit(NULL_MESSAGE)


def test_null_channel():
    ch = Channel()
    assert ch.transmit(NULL_MESSAGE).is_null
    with pytest.raises(ValueError):
        ch.transmit(Message(np.zeros(1)))


def test_registry():
    @register("server", "_test_first")
    class First(ServerPolicy):
        def act(self, obs):
            return ServerAction(0)

    assert "_test_first" in available("server")
    p = make("server", "_test_first")
    p.reset(1)
    assert isinstance(p, First)
    with pytest.raises(TypeError):
        register("switch", "_bad")(First)
    with pytest.raises(KeyError):
        make("switch", "_nope")
    assert issubclass(SwitchPolicy, object)


def test_estimators():
    reps = (ServerReport(np.zeros((1, 2)), 0.0, 1.0),)
    assert make_estimator("none").estimate(reps, 1.0) is None
    assert make_estimator("last_report").estimate(reps, 1.0)[0].shape == (1, 2)
    with pytest.raises(KeyError):
        make_estimator("projection")
