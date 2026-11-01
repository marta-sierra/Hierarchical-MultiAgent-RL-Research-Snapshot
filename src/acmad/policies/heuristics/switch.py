"""Switch heuristics on what the switch observes: aged ACK reports, c_t and the topology (ADR-0014)."""
from __future__ import annotations

import numpy as np

from acmad.interface import SUMMARY_FIELDS, SwitchAction, SwitchObs, SwitchPolicy, register

_BUSY, _QUEUED = SUMMARY_FIELDS.index("n_busy"), SUMMARY_FIELDS.index("n_queued")
TASKS_PER_JOB = 2      # cpu_processing -> ldpc_decoding (ADR-0011)


@register("switch", "least_loaded")
class LeastLoaded(SwitchPolicy):
    """Server with the fewest tasks per unit of capacity, as the switch sees it:

        load_k = (sum_e n_busy + n_queued in k's newest report  +  TASKS_PER_JOB * jobs sent since it)
                 / sum_e capacity_e

    The report is taken as is (`last_report`, ADR-0014) and corrected by c_t's dispatch record. CPU
    cores and GPUs count as one unit each. Ties: uniform among the tied servers (seeded rng).
    """

    def act(self, obs: SwitchObs) -> SwitchAction:
        in_system = np.array([r.summary[:, _BUSY].sum() + r.summary[:, _QUEUED].sum() for r in obs.reports])
        units = np.array([sum(e.capacity for e in srv) for srv in obs.topology], dtype=float)
        load = (in_system + TASKS_PER_JOB * obs.cluster.jobs_since_ack) / units
        best = np.flatnonzero(load == load.min())
        return SwitchAction(int(best[self.rng.integers(len(best))]))
