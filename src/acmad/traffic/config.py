"""`TrafficConfig`: the resolved, frozen config of the traffic generator (docs/TRAFFIC.md §13)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "traffic" / "default.yaml"

GENERATOR_VERSION = "0.8"       # part of every cache key: bump whenever a stage's output changes
TTI_MS = 1.0                    # LTE TTI of the trace
MCS_TABLE = "38.214-T2"
SPLITS = ("train", "val", "test")
LOAD_CLASSES = ("low", "medium", "high")    # ADR-0007: super-block load class, from the strata of A7


@dataclass(frozen=True)
class TraceConfig:
    path: str = "TrafficGenerator/traces.csv"
    drop_rntis: tuple[int, ...] = (65534, 65535)
    codeword_mcs: str = "max"
    retransmissions: str = "drop"


@dataclass(frozen=True)
class SplitConfig:
    block_s: int = 20
    superblock_s: int = 120
    split_seed: int = 0
    stratum_size: int = 5
    per_stratum: tuple[tuple[str, int], ...] = (("train", 3), ("val", 1), ("test", 1))

    @property
    def block_ms(self) -> int:
        return self.block_s * 1000

    @property
    def blocks_per_superblock(self) -> int:
        return self.superblock_s // self.block_s


@dataclass(frozen=True)
class RadioConfig:
    slot_ms: float = 0.5
    carrier_prb: int = 273
    pusch_prb: int = 250
    re_per_prb: int = 156
    layers: int = 1
    mcs_table: str = MCS_TABLE
    lte_to_nr_mcs: str = "B"
    buffer_expiry_slots: int = 4

    @property
    def slots_per_tti(self) -> int:
        return int(round(TTI_MS / self.slot_ms))


@dataclass(frozen=True)
class ChannelConfig:
    cloudric_dir: str = "data/cloudric"
    snr_decay_db: float = 1.0     # ADR-0020: P(snr_min + k) ∝ exp(-k / decay); author's decision


@dataclass(frozen=True)
class TimingConfig:
    k2_slots: int = 4
    fronthaul_ms: float = 0.0
    jitter_ms: float = 0.0


@dataclass(frozen=True)
class LoadConfig:
    n_dus: int = 1
    s: float = 1.0


@dataclass(frozen=True)
class CacheConfig:
    dir: str = "data/cache"


@dataclass(frozen=True)
class TrafficConfig:
    """Everything the generator reads, besides the per-episode `(split, n_dus, s, seed)`."""

    trace: TraceConfig = field(default_factory=TraceConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    radio: RadioConfig = field(default_factory=RadioConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    load: LoadConfig = field(default_factory=LoadConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    def __post_init__(self) -> None:
        _check(self)

    # ---- paths (relative ones are resolved against the repo root) ----
    @property
    def trace_path(self) -> Path:
        return _resolve(self.trace.path)

    @property
    def cloudric_dir(self) -> Path:
        return _resolve(self.channel.cloudric_dir)

    @property
    def cache_dir(self) -> Path:
        return _resolve(self.cache.dir)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["split"]["per_stratum"] = dict(self.split.per_stratum)
        d["trace"]["drop_rntis"] = list(self.trace.drop_rntis)
        return d

    def hash(self) -> str:
        """Content hash of the resolved config (paths as written, not their contents)."""
        return _key(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrafficConfig":
        known = {f.name: f.type for f in fields(cls)}
        unknown = set(d) - set(known)
        if unknown:
            raise ValueError(f"unknown traffic config sections {sorted(unknown)}")
        sections = {}
        for f in fields(cls):
            sub = dict(d.get(f.name) or {})
            sub_cls = f.default_factory                                    # type: ignore[misc]
            allowed = {g.name for g in fields(sub_cls)}
            if set(sub) - allowed:
                raise ValueError(f"unknown keys in traffic.{f.name}: {sorted(set(sub) - allowed)}")
            if "drop_rntis" in sub:
                sub["drop_rntis"] = tuple(int(x) for x in sub["drop_rntis"])
            if "per_stratum" in sub:
                sub["per_stratum"] = tuple((str(k), int(v)) for k, v in dict(sub["per_stratum"]).items())
            sections[f.name] = sub_cls(**sub)
        return cls(**sections)

    @classmethod
    def from_yaml(cls, path: str | Path = DEFAULT_CONFIG) -> "TrafficConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})


def load_config(path: str | Path = DEFAULT_CONFIG) -> TrafficConfig:
    """Read a traffic YAML (default: `configs/traffic/default.yaml`)."""
    return TrafficConfig.from_yaml(path)


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else REPO_ROOT / q


def _key(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _check(c: TrafficConfig) -> None:
    """Only the decided choices are implemented (docs/adr/); refuse anything else."""
    if c.trace.codeword_mcs != "max":
        raise ValueError("trace.codeword_mcs: only 'max' is implemented (ADR-0018)")
    if c.trace.retransmissions != "drop":
        raise ValueError("trace.retransmissions: only 'drop' is implemented (ADR-0018)")
    if c.radio.mcs_table != MCS_TABLE:
        raise ValueError(f"radio.mcs_table: only {MCS_TABLE} is implemented (ADR-0019)")
    if c.radio.lte_to_nr_mcs != "B":
        raise ValueError("radio.lte_to_nr_mcs: only option 'B' is implemented (ADR-0018)")
    if c.radio.buffer_expiry_slots < 1:
        raise ValueError("radio.buffer_expiry_slots must be >= 1 (ADR-0019: expiry always on)")
    if c.split.superblock_s % c.split.block_s:
        raise ValueError("split.superblock_s must be a multiple of split.block_s")
    if sum(n for _, n in c.split.per_stratum) != c.split.stratum_size:
        raise ValueError("split.per_stratum must add up to split.stratum_size")
    if set(dict(c.split.per_stratum)) != set(SPLITS):
        raise ValueError(f"split.per_stratum must name exactly {SPLITS}")
    if abs(TTI_MS / c.radio.slot_ms - round(TTI_MS / c.radio.slot_ms)) > 1e-12:
        raise ValueError("radio.slot_ms must divide the 1 ms TTI")
    if not c.channel.snr_decay_db >= 0:
        raise ValueError("channel.snr_decay_db must be >= 0 (ADR-0020)")
    if c.radio.pusch_prb > c.radio.carrier_prb:
        raise ValueError("radio.pusch_prb > radio.carrier_prb")
