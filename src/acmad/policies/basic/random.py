"""Uniformly random choice for both tiers, from the policy's own seeded rng."""
from __future__ import annotations

from acmad.interface import ServerAction, ServerObs, ServerPolicy, SwitchAction, SwitchObs, SwitchPolicy, register


@register("switch", "random")
class RandomSwitch(SwitchPolicy):
    """Server k uniform over the K servers, null message."""

    def act(self, obs: SwitchObs) -> SwitchAction:
        return SwitchAction(int(self.rng.integers(len(obs.reports))))


@register("server", "random")
class RandomServer(ServerPolicy):
    """Executor e uniform over the allowed executors of the server."""

    def act(self, obs: ServerObs) -> ServerAction:
        ok = [e for e, x in enumerate(obs.executors) if x.allowed]
        return ServerAction(int(ok[self.rng.integers(len(ok))]))
