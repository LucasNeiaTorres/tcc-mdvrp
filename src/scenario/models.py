"""Shared scenario model types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureEvent:
    trigger_time: float
    type: str
    node_a: int
    node_b: int
