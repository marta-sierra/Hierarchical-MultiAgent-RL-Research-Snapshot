"""Dashboard backend: a recorded episode runs end to end and its invariants hold."""
import pytest

from acmad.dashboard.episode import options, run_episode
from acmad.traffic.config import load_config

pytestmark = pytest.mark.trace


@pytest.fixture(scope="module", autouse=True)
def _data():
    c = load_config()
    if not c.trace_path.exists() or not (c.cloudric_dir / "CPU_dataset.csv").exists():
        pytest.skip("real trace or CloudRIC data not available")


@pytest.mark.parametrize("kw", [{"horizon_ms": 30},
                                {"horizon_ms": 30, "deadline_ms": 0.5, "downlink_ms": 0.05, "uplink_ms": 0.1}])
def test_episode_checks_pass(kw):
    ep = run_episode(kw)
    assert ep["tasks"] and all(c["ok"] for c in ep["checks"]), [c for c in ep["checks"] if not c["ok"]]
    assert sum(ep["outcomes"].values()) == len(ep["tasks"])
    for t in ep["tasks"]:                     # stages are entered in event order
        idx = list(t["stage"].values())
        assert idx == sorted(idx) and "end" in t["stage"]


def test_everything_dropped_in_flight():
    ep = run_episode({"horizon_ms": 10, "deadline_ms": 0.1, "downlink_ms": 0.2})
    assert ep["outcomes"] == {"dropped": len(ep["tasks"])}
    assert all(set(t["stage"]) == {"arrival", "dispatch", "end"} for t in ep["tasks"])


def test_options_lists_topologies_and_policies():
    o = options()
    assert "example" in o["topologies"] and "round_robin" in o["policies"]["switch"]
