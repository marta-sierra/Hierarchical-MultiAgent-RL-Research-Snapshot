"""Downlink header channel: message dimension and b-bit quantisation (ARCHITECTURE §6).

Matched bandwidth: `preference`, `goal` and `latent` channels share `dim` and `bits`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acmad.interface.actions import NULL_MESSAGE, Message


@dataclass(frozen=True)
class Channel:
    """A header field of `dim` values, each quantised uniformly to `bits` bits on [low, high].

    `dim == 0` is the null channel: only the null message is accepted.
    `bits is None` means no quantisation (full float64).
    """

    dim: int = 0
    bits: int | None = None
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError("dim must be >= 0")
        if self.bits is not None and not 1 <= self.bits <= 32:
            raise ValueError("bits must be in 1..32 or None")
        if not self.high > self.low:
            raise ValueError("need high > low")

    @property
    def is_null(self) -> bool:
        return self.dim == 0

    @property
    def header_bits(self) -> int:
        """Total downlink bits per dispatch (64 per value if unquantised)."""
        return self.dim * (64 if self.bits is None else self.bits)

    def validate(self, msg: Message) -> None:
        """Raise `ValueError` if `msg` does not fit this channel."""
        if self.is_null:
            if not msg.is_null:
                raise ValueError("null channel carries only the null message")
            return
        if msg.is_null:
            raise ValueError(f"channel of dim {self.dim} needs a non-null message")
        v = msg.vector
        if v.shape != (self.dim,):
            raise ValueError(f"message shape {v.shape} != ({self.dim},)")
        if not np.all(np.isfinite(v)):
            raise ValueError("message has non-finite values")

    def quantise(self, v: np.ndarray) -> np.ndarray:
        """Clip to [low, high] and round to the nearest of 2**bits uniform levels."""
        v = np.clip(np.asarray(v, dtype=np.float64), self.low, self.high)
        if self.bits is None:
            return v
        levels = (1 << self.bits) - 1
        q = np.round((v - self.low) / (self.high - self.low) * levels)
        return self.low + q * (self.high - self.low) / levels

    def transmit(self, msg: Message) -> Message:
        """What the server receives: validated and quantised."""
        self.validate(msg)
        return NULL_MESSAGE if self.is_null else Message(self.quantise(msg.vector))


NULL_CHANNEL = Channel(dim=0)
