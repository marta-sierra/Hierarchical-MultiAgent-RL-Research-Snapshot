"""Blocked set of a server: tasks waiting for a predecessor (ADR-0011)."""
from __future__ import annotations


class BlockedSet:
    """Not FIFO: a task leaves as soon as all its predecessors have completed.

    TaskStream gives at most one predecessor per task (`depends_on`), but the tracker
    accepts any number.
    """

    def __init__(self) -> None:
        self._waiting_on: dict[int, set[int]] = {}     # task -> unfinished predecessors
        self._children: dict[int, list[int]] = {}      # predecessor -> tasks waiting on it
        self._done: set[int] = set()

    def __contains__(self, task_id: int) -> bool:
        return task_id in self._waiting_on

    def __len__(self) -> int:
        return len(self._waiting_on)

    def add(self, task_id: int, predecessors: tuple[int, ...]) -> bool:
        """Register a task; returns True if it is ready now (no unfinished predecessor)."""
        pending = {p for p in predecessors if p not in self._done}
        if not pending:
            return True
        self._waiting_on[task_id] = pending
        for p in pending:
            self._children.setdefault(p, []).append(task_id)
        return False

    def complete(self, task_id: int) -> list[int]:
        """Mark `task_id` done; returns the tasks released, in registration order."""
        self._done.add(task_id)
        released = []
        for c in self._children.pop(task_id, []):
            w = self._waiting_on.get(c)
            if w is None:
                continue
            w.discard(task_id)
            if not w:
                del self._waiting_on[c]
                released.append(c)
        return released

    def remove(self, task_id: int) -> None:
        """Drop a blocked task (deadline, ADR-0013)."""
        for p in self._waiting_on.pop(task_id, ()):
            kids = self._children.get(p)
            if kids is not None and task_id in kids:
                kids.remove(task_id)
