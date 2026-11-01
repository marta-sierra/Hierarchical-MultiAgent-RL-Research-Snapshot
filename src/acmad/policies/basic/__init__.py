"""Basic baseline policies: no state is read, only the order of decisions or a seeded rng."""
from acmad.policies.basic.random import RandomServer, RandomSwitch
from acmad.policies.basic.round_robin import RoundRobinServer, RoundRobinSwitch

__all__ = ["RandomSwitch", "RandomServer", "RoundRobinSwitch", "RoundRobinServer"]
