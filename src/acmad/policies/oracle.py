"""Server policies of the explicit service-oracle mode (ADR-0014): they see the exact service time and
dynamic energy of the offered task on each executor (`ServerObs.oracle_service`). Upper-bound
references, not deployable (ADR-0001). Ported from the old code's `fastest`, `cheapest` and the
switch environment's `_select_best_processor` (weighted score)."""
from __future__ import annotations

from acmad.interface import ServerAction, ServerObs, ServerPolicy, register


class _ServiceOracle(ServerPolicy):
    REQUIRES = frozenset({"oracle_service"})

    def score(self, service_time_ms: float, dynamic_energy_j: float) -> float:
        raise NotImplementedError

    def act(self, obs: ServerObs) -> ServerAction:
        if obs.oracle_service is None:
            raise RuntimeError(f"{type(self).__name__} needs the service oracle (sim.oracle_service: true)")
        cand = [(self.score(*c), e) for e, c in enumerate(obs.oracle_service) if c is not None]
        return ServerAction(min(cand)[1])


@register("server", "fastest")
class Fastest(_ServiceOracle):
    """Executor with the shortest service time of this task (queues ignored)."""

    def score(self, service_time_ms, dynamic_energy_j):
        return service_time_ms


@register("server", "cheapest")
class Cheapest(_ServiceOracle):
    """Executor with the lowest dynamic energy of this task (queues ignored)."""

    def score(self, service_time_ms, dynamic_energy_j):
        return dynamic_energy_j


@register("server", "weighted_score")
class WeightedScore(_ServiceOracle):
    """alpha * sigma [ms] + (1 - alpha) * energy [mJ] (old `_select_best_processor`; queues ignored)."""

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)

    def score(self, service_time_ms, dynamic_energy_j):
        return self.alpha * service_time_ms + (1.0 - self.alpha) * dynamic_energy_j * 1e3
