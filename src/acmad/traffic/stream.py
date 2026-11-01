"""`TaskStream`: the only output of `acmad.traffic` and the only traffic input of `acmad.sim`.

Schema: docs/TRAFFIC.md §10 (ADR-0016, ADR-0012). Times are in ms (float64).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

JOB_COLUMNS: dict[str, str] = {"job_id": "int64", "du": "int16", "t_arrival_ms": "float64"}
TASK_COLUMNS: dict[str, str] = {
    "task_id": "int64", "job_id": "int64", "kind": "category", "allowed": "category",
    "depends_on": "Int64", "nprb": "int16", "mcs": "int8", "snr_db": "int8", "tbs": "int32",
    # fixed by (mcs, nprb) per TS 38.214 / 38.212 (ADR-0019): LDPC segmentation and modulation
    "base_graph": "int8", "n_cb": "int16", "cb_len": "int32", "total_bits": "int32", "qm": "int8",
    "code_rate": "float32",
}
TASK_KINDS = ("cpu_processing", "ldpc_decoding")


@dataclass(frozen=True)
class TaskStream:
    """One episode of traffic: `jobs` (one row per grant chain), `tasks` (two per job), `meta`."""

    jobs: pd.DataFrame
    tasks: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Raise `ValueError` if the tables do not follow the schema."""
        for name, df, cols in (("jobs", self.jobs, JOB_COLUMNS), ("tasks", self.tasks, TASK_COLUMNS)):
            got = {c: str(t) for c, t in df.dtypes.items()}
            if got != cols:
                raise ValueError(f"{name} schema {got} != {cols}")
        if not self.jobs.t_arrival_ms.is_monotonic_increasing:
            raise ValueError("jobs must be sorted by t_arrival_ms")
        if not set(self.tasks.kind.unique()) <= set(TASK_KINDS):
            raise ValueError(f"unknown task kinds {set(self.tasks.kind.unique()) - set(TASK_KINDS)}")
        if not self.tasks.job_id.isin(self.jobs.job_id).all():
            raise ValueError("task with a job_id not in jobs")

    def save(self, path: str | Path) -> None:
        """Write `jobs.parquet`, `tasks.parquet` and `meta.json` into directory `path`."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.jobs.to_parquet(path / "jobs.parquet", compression="zstd", index=False)
        self.tasks.to_parquet(path / "tasks.parquet", compression="zstd", index=False)
        (path / "meta.json").write_text(json.dumps(self.meta, indent=1, sort_keys=True, default=_json_default))

    @classmethod
    def load(cls, path: str | Path) -> "TaskStream":
        path = Path(path)
        return cls(pd.read_parquet(path / "jobs.parquet"), pd.read_parquet(path / "tasks.parquet"),
                   json.loads((path / "meta.json").read_text()))

    def content_hash(self) -> str:
        """Hash of the tables' content (not of `meta`), for records and determinism tests."""
        h = hashlib.sha256()
        for df in (self.jobs, self.tasks):
            h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
        return h.hexdigest()[:16]


def _json_default(x):
    if hasattr(x, "item"):          # numpy scalars
        return x.item()
    raise TypeError(f"not JSON serialisable: {type(x)}")
