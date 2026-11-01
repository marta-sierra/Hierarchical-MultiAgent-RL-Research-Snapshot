"""Round robin for both tiers: cycle 0, 1, ..., n-1, 0, ... in decision order."""
from __future__ import annotations

from acmad.interface import ServerAction, ServerObs, ServerPolicy, SwitchAction, SwitchObs, SwitchPolicy, register


@register("switch", "round_robin")
class RoundRobinSwitch(SwitchPolicy):
    """Job j-th dispatched goes to server k = j mod K (K = number of servers), null message."""

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._next = 0

    def act(self, obs: SwitchObs) -> SwitchAction:
        k = self._next % len(obs.reports)
        self._next = k + 1
        return SwitchAction(k)


@register("server", "round_robin")
class RoundRobinServer(ServerPolicy):
    """Each server cycles over its executors e in E_k, skipping those not allowed for the task.

    One cursor per `server_id`, so a single instance can serve every server.
    """

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._next: dict[int, int] = {}

    def act(self, obs: ServerObs) -> ServerAction:
        n = len(obs.executors)
        start = self._next.get(obs.server_id, 0)
        for i in range(n):
            e = (start + i) % n
            if obs.executors[e].allowed:
                self._next[obs.server_id] = e + 1
                return ServerAction(e)
        raise ValueError(f"no allowed executor on server {obs.server_id} for task {obs.task.id}")
