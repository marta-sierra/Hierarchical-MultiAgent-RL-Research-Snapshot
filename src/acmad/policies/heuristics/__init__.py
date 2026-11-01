"""Deterministic heuristic policies on the observable state (no service times, ADR-0001)."""
from acmad.policies.heuristics.server import AllCPU, AllGPU, LeastQueued
from acmad.policies.heuristics.switch import LeastLoaded

__all__ = ["LeastLoaded", "AllCPU", "AllGPU", "LeastQueued"]
