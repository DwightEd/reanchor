"""Label-free provenance partitions for reanchor mechanism audits.

The four coarse groups are bookkeeping coordinates, not discovered mechanism
classes.  Material ``source_unit_id`` values retain the finer identity needed
to ask which document/record, rather than which broad bucket, carried an effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCE_GROUPS = ("constraint", "content", "other_prompt", "response_history")


@dataclass(frozen=True)
class SourcePartition:
    """Token masks that exactly partition ordinary prompt/history sources."""

    names: tuple[str, ...]
    masks: np.ndarray
    semantic_roles_available: bool

    def __post_init__(self) -> None:
        masks = np.asarray(self.masks)
        if masks.ndim != 2 or masks.shape[0] != len(self.names):
            raise ValueError("source masks must have shape [group, token]")
        if masks.dtype != np.bool_:
            raise ValueError("source masks must be boolean")
        if np.any(masks.sum(0) > 1):
            raise ValueError("source groups must be disjoint")


@dataclass(frozen=True)
class SourceUnitPartition:
    """Material source units and their coarse provenance roles."""

    ids: np.ndarray
    roles: tuple[str, ...]
    masks: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.ids)
        masks = np.asarray(self.masks)
        if ids.ndim != 1 or masks.ndim != 2 or masks.shape[0] != len(ids):
            raise ValueError("source-unit masks must have shape [unit, token]")
        if len(self.roles) != len(ids) or masks.dtype != np.bool_:
            raise ValueError("source-unit metadata axes disagree")
        if len(set(map(int, ids))) != len(ids) or np.any(ids < 0):
            raise ValueError("source-unit IDs must be unique and nonnegative")
        if np.any(masks.sum(0) > 1):
            raise ValueError("source units must be disjoint")


def source_partition(trace: dict[str, np.ndarray]) -> SourcePartition:
    """Build a complete source partition without consulting outcome labels.

    A capture may provide ``source_kind`` with token-level values ``constraint``,
    ``content`` or ``other``. Older v3 captures fall back to ``evidence_mask``:
    evidence becomes content and the constraint group is explicitly unavailable.
    Generated response history always takes precedence over prompt annotations.
    """

    token_count = len(trace["token_ids"])
    special = _boolean_field(trace, "special_mask", token_count)
    ordinary = ~special
    response_start = int(np.asarray(trace["response_start"]))
    if not 0 <= response_start <= token_count:
        raise ValueError("response_start lies outside token_ids")
    positions = np.arange(token_count)
    prompt = positions < response_start
    history = ordinary & ~prompt

    semantic_roles_available = "source_kind" in trace
    if semantic_roles_available:
        kinds = np.asarray(trace["source_kind"]).astype(str)
        if kinds.shape != (token_count,):
            raise ValueError("source_kind must have one value per token")
        unknown = sorted(set(kinds[ordinary & prompt]) - {"", "constraint", "content", "other"})
        if unknown:
            raise ValueError(f"unsupported source_kind values: {unknown[:5]}")
        evidence = _boolean_field(trace, "evidence_mask", token_count)
        constraint = ordinary & prompt & (kinds == "constraint")
        content = ordinary & prompt & ((kinds == "content") | ((kinds == "") & evidence))
    else:
        evidence = _boolean_field(trace, "evidence_mask", token_count)
        constraint = np.zeros(token_count, dtype=bool)
        content = ordinary & prompt & evidence

    other_prompt = ordinary & prompt & ~constraint & ~content
    masks = np.stack((constraint, content, other_prompt, history))
    if not np.array_equal(masks.any(0), ordinary):
        raise ValueError("source groups do not cover every ordinary token")
    return SourcePartition(SOURCE_GROUPS, masks, semantic_roles_available)


def source_unit_partition(
    trace: dict[str, np.ndarray], coarse: SourcePartition | None = None
) -> SourceUnitPartition:
    """Retain every annotated material unit instead of averaging it away.

    ``-1`` is the capture convention for tokens without a material unit.  Those
    tokens remain represented by the complete coarse partition but are not
    invented as semantic source units here.
    """

    token_count = len(trace["token_ids"])
    if "source_unit_id" not in trace:
        return SourceUnitPartition(
            np.empty(0, dtype=np.int64), (), np.zeros((0, token_count), dtype=bool)
        )
    unit_ids = np.asarray(trace["source_unit_id"], dtype=np.int64)
    if unit_ids.shape != (token_count,):
        raise ValueError("source_unit_id must have one value per token")
    ordinary = ~_boolean_field(trace, "special_mask", token_count)
    ids = np.asarray(sorted(set(unit_ids[ordinary & (unit_ids >= 0)])), dtype=np.int64)
    masks = (
        np.stack([ordinary & (unit_ids == value) for value in ids])
        if len(ids)
        else np.zeros((0, token_count), dtype=bool)
    )
    coarse = coarse or source_partition(trace)
    roles = []
    for mask in masks:
        matches = [
            name
            for name, group in zip(coarse.names, coarse.masks, strict=True)
            if np.any(mask & group)
        ]
        if not matches:
            raise ValueError("source unit is absent from the coarse source partition")
        roles.append(matches[0] if len(matches) == 1 else "mixed:" + "+".join(matches))
    return SourceUnitPartition(ids, tuple(roles), masks)


def _boolean_field(trace: dict[str, np.ndarray], name: str, size: int) -> np.ndarray:
    if name not in trace:
        if name == "evidence_mask":
            return np.zeros(size, dtype=bool)
        raise ValueError(f"capture is missing {name}")
    value = np.asarray(trace[name], dtype=bool)
    if value.shape != (size,):
        raise ValueError(f"{name} must have one value per token")
    return value
