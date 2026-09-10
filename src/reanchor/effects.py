"""Factorial effects for one constraint-control event."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BranchMargins:
    """A-minus-B margins before and after forced commitments."""

    onset_a: float
    onset_b: float
    world_a_after_a: float
    world_a_after_b: float
    world_b_after_a: float
    world_b_after_b: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ControlEffects:
    """Four A/B-invariant coordinates of source and prefix control."""

    source_onset: float
    source_followup: float
    prefix_followup: float
    source_prefix_coupling: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_effects(margins: BranchMargins) -> ControlEffects:
    """Decompose the source-by-prefix factorial design without mixing effects."""

    source_followup = 0.25 * (
        margins.world_a_after_a
        + margins.world_a_after_b
        - margins.world_b_after_a
        - margins.world_b_after_b
    )
    prefix_followup = 0.25 * (
        margins.world_a_after_a
        - margins.world_a_after_b
        + margins.world_b_after_a
        - margins.world_b_after_b
    )
    interaction = 0.25 * (
        margins.world_a_after_a
        - margins.world_a_after_b
        - margins.world_b_after_a
        + margins.world_b_after_b
    )
    return ControlEffects(
        source_onset=0.5 * (margins.onset_a - margins.onset_b),
        source_followup=source_followup,
        prefix_followup=prefix_followup,
        source_prefix_coupling=abs(interaction),
    )
