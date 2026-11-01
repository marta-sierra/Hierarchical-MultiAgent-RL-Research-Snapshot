"""Policy ABCs for the two tiers and a by-name registry (ARCHITECTURE §4).

Policies expose only `reset(seed)` and `act(obs)`; learning lives in `training/`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Literal, TypeVar

import numpy as np

from acmad.interface.actions import ServerAction, SwitchAction
from acmad.interface.observations import ServerObs, SwitchObs

Tier = Literal["switch", "server"]


class _Policy(ABC):
    rng: np.random.Generator
    #: Explicit observation modes the policy needs (`"oracle_cluster"`, `"oracle_service"`, ADR-0014);
    #: the experiment runner switches them on in the simulator.
    REQUIRES: frozenset[str] = frozenset()

    def reset(self, seed: int) -> None:
        """Start an episode; the policy's only randomness is `self.rng`."""
        self.rng = np.random.default_rng(seed)


class SwitchPolicy(_Policy):
    """Brain of the upstream dispatcher u: job -> (server k, message w_i)."""

    @abstractmethod
    def act(self, obs: SwitchObs) -> SwitchAction: ...


class ServerPolicy(_Policy):
    """Brain of a downstream dispatcher k: ready task -> executor e."""

    @abstractmethod
    def act(self, obs: ServerObs) -> ServerAction: ...


_BASES: dict[str, type] = {"switch": SwitchPolicy, "server": ServerPolicy}
_REGISTRY: dict[str, dict[str, type]] = {"switch": {}, "server": {}}
P = TypeVar("P", bound=type)


def register(tier: Tier, name: str) -> Callable[[P], P]:
    """Class decorator: make a policy selectable from config as `(tier, name)`."""
    if tier not in _BASES:
        raise ValueError(f"unknown tier {tier!r}")

    def deco(cls: P) -> P:
        if not issubclass(cls, _BASES[tier]):
            raise TypeError(f"{cls.__name__} is not a {_BASES[tier].__name__}")
        prev = _REGISTRY[tier].get(name)
        if prev is not None and prev is not cls:
            raise ValueError(f"{tier} policy {name!r} already registered by {prev.__name__}")
        _REGISTRY[tier][name] = cls
        return cls

    return deco


def make(tier: Tier, name: str, **kwargs) -> SwitchPolicy | ServerPolicy:
    """Instantiate a registered policy."""
    try:
        cls = _REGISTRY[tier][name]
    except KeyError:
        raise KeyError(f"no {tier} policy {name!r}; available: {available(tier)}") from None
    return cls(**kwargs)


def available(tier: Tier) -> tuple[str, ...]:
    """Registered policy names for a tier, sorted."""
    return tuple(sorted(_REGISTRY[tier]))
