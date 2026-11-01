"""Server k (mechanics): blocked set, ready queue, executors (ADR-0011, ADR-0010)."""
from __future__ import annotations

from collections import deque

from acmad.sim.dependencies import BlockedSet
from acmad.sim.executor import Executor
from acmad.sim.rules import cpu_index


class Server:
    """Tasks arrive (whole jobs, ADR-0011), wait in the blocked set until their predecessors
    complete, then join the ready queue tail. The engine drains the ready queue at once:
    each task is bound to an executor (by rule or by the server policy) and waits in
    that executor's FIFO.
    """

    def __init__(self, server_id: int, executors: list[Executor]) -> None:
        self.id = server_id
        self.executors = executors
        self.cpu = cpu_index([e.spec for e in executors])
        self.blocked = BlockedSet()
        self.ready: deque[int] = deque()

    def receive(self, task_ids: list[int], predecessors: dict[int, tuple[int, ...]]) -> list[int]:
        """Admit tasks; returns those that entered the ready queue (in order)."""
        entered = []
        for t in task_ids:
            if self.blocked.add(t, predecessors.get(t, ())):
                self.ready.append(t)
                entered.append(t)
        return entered

    def complete(self, task_id: int) -> list[int]:
        """A task finished; move released successors to the ready-queue tail."""
        released = self.blocked.complete(task_id)
        self.ready.extend(released)
        return released
