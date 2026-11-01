"""Server heuristics on the observable local state (ADR-0001): no service time or energy is read."""
from __future__ import annotations

from acmad.interface import ServerAction, ServerObs, ServerPolicy, register


def _allowed(obs: ServerObs) -> list[int]:
    ok = [e for e, x in enumerate(obs.executors) if x.allowed]
    if not ok:
        raise ValueError(f"no allowed executor on server {obs.server_id} for task {obs.task.id}")
    return ok


class _PreferKind(ServerPolicy):
    """Always the first allowed executor of kind `KIND`; the first allowed one if the server has none
    (e.g. `all_gpu` on a GPU-less server runs on its CPU)."""

    KIND: str

    def act(self, obs: ServerObs) -> ServerAction:
        ok = _allowed(obs)
        return ServerAction(next((e for e in ok if obs.executors[e].kind == self.KIND), ok[0]))


@register("server", "all_cpu")
class AllCPU(_PreferKind):
    KIND = "cpu"


@register("server", "all_gpu")
class AllGPU(_PreferKind):
    KIND = "gpu"


@register("server", "least_queued")
class LeastQueued(ServerPolicy):
    """Allowed executor with the fewest tasks per unit, (n_busy + n_queued) / capacity; ties to the
    lowest index (on an empty server: the first executor, the CPU in the reference topology)."""

    def act(self, obs: ServerObs) -> ServerAction:
        x = obs.executors
        return ServerAction(min(_allowed(obs), key=lambda e: ((x[e].n_busy + x[e].n_queued) / x[e].capacity, e)))
