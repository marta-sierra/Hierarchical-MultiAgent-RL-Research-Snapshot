"""Closed block: real 5G traces -> TaskStream (docs/TRAFFIC.md). Only this package reads raw traces.

Stages: `traces` (A: trace prep + split + load classes), `radio` (R: 3GPP / CloudRIC tables), `scaling`
(B0: episode), `workload` (B1-B3: requests, scheduler, reception), `api` (`generate`).
"""
from .api import generate, generate_from, load_or_generate, stream_dir
from .config import GENERATOR_VERSION, LOAD_CLASSES, TrafficConfig, load_config
from .stream import TaskStream

__all__ = ["GENERATOR_VERSION", "LOAD_CLASSES", "TaskStream", "TrafficConfig", "generate", "generate_from", "load_config",
           "load_or_generate", "stream_dir"]
