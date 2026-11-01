"""Fixtures for the traffic tests. Tests that need the real trace are marked `trace` and skip without it."""
from __future__ import annotations

import pytest

from acmad.traffic.config import load_config


def pytest_configure(config):
    config.addinivalue_line("markers", "trace: needs the real trace (TrafficGenerator/traces.csv) and data/cloudric/")


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    if not c.trace_path.exists() or not (c.cloudric_dir / "CPU_dataset.csv").exists():
        pytest.skip("real trace or CloudRIC data not available")
    return c


@pytest.fixture(scope="session")
def prep(cfg):
    from acmad.traffic.traces import ensure_prep
    return ensure_prep(cfg)


@pytest.fixture(scope="session")
def radio(cfg, prep):
    from acmad.traffic.radio import ensure_radio
    return ensure_radio(cfg, prep)


@pytest.fixture(scope="session")
def cloudric(cfg):
    from acmad.cloudric import read_cloudric
    return read_cloudric(cfg.cloudric_dir)
