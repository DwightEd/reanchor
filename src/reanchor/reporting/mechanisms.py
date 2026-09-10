"""Outcome-joined measurements for transition mechanism traces."""

from __future__ import annotations

import numpy as np

from .statistics import received_then_overridden

MECHANISM_SCORES = (
    "baseline_margin",
    "transition_seed_norm",
    "transition_remote_effect",
    "transition_margin_per_seed_norm",
    "transition_nonadoption_score",
    "transition_multi_hop_effect",
    "current_remote_write_effect",
    "linearized_margin_without_current_remote_write",
    "source_group_effect_cancellation",
    "constraint_transition_coefficient_l1_share",
    "content_transition_coefficient_l1_share",
    "response_history_transition_coefficient_l1_share",
)


def mechanism_target_row(
    artifact: dict[str, np.ndarray],
    target_index: int,
    *,
    identity: dict,
    label: int,
    target_phase: str,
    before_next_event: bool,
    effect_epsilon: float = 1e-6,
) -> dict:
    """Return continuous route and margin effects for one generated target.

    Coarse source groups are deliberately not converted into named hallucination
    mechanisms. The event/target phase and exact hop decomposition are the main
    axes; provenance is an interpretation aid.
    """

    names = tuple(map(str, artifact["source_group_names"]))
    margins = np.asarray(artifact["source_group_margin_response"], dtype=float)
    layers = np.asarray(artifact["source_group_layer_margin_response"], dtype=float)
    roots = np.asarray(artifact["source_group_root_coefficient"], dtype=float)
    seed_norms = np.asarray(artifact["source_group_seed_norm"], dtype=float)
    if margins.shape[0] != len(names) or layers.shape[0] != len(names):
        raise ValueError("source group axes disagree")
    if roots.shape[0] != len(names) or seed_norms.shape != (len(names),):
        raise ValueError("source group diagnostics disagree")

    effects = margins[:, 0, :, target_index].sum(1)
    layer_effects = layers[:, 0, :, :, target_index].sum(1)
    signed_coefficients = roots.sum(1)
    coefficient_l1 = np.abs(roots).sum(1)
    total_l1 = float(coefficient_l1.sum())
    l1_shares = coefficient_l1 / total_l1 if total_l1 > 0 else np.zeros_like(coefficient_l1)
    transition = np.asarray(artifact["transition_margin_response"], dtype=float)
    transition_hops = transition[0, :, target_index]
    transition_effect = float(transition_hops.sum())
    transition_seed_norm = float(np.asarray(artifact["transition_seed_norm"]).item())
    effect_per_seed = (
        transition_effect / transition_seed_norm
        if transition_seed_norm > effect_epsilon
        else np.nan
    )
    current = np.asarray(artifact["current_remote_margin_response"], dtype=float)
    current_hops = current[0, :, target_index]
    current_effect = float(current_hops.sum())
    baseline = float(np.asarray(artifact["baseline_margin"])[target_index])
    closure_pass = bool(artifact["closure_pass"])
    group_effect_l1 = float(np.abs(effects).sum())
    cancellation = 1.0 - abs(transition_effect) / group_effect_l1 if group_effect_l1 > 0 else 0.0
    event_position = int(identity["event_position"])
    target_position = int(artifact["target_position"][target_index])
    event_target_offset = target_position - event_position
    onset_aligned = target_phase == "onset" and event_target_offset == 1
    rollout_target = target_phase == "continuing" and event_target_offset > 1 and before_next_event
    rollout_multi_hop_negative = bool(
        closure_pass and rollout_target and transition_hops[2] < -effect_epsilon
    )
    row = {
        **identity,
        "target_position": target_position,
        "event_target_offset": event_target_offset,
        "target_phase": target_phase,
        "before_next_event": bool(before_next_event),
        "onset_aligned": onset_aligned,
        "rollout_target": rollout_target,
        "label": int(label),
        "readout": (
            "explicit_candidate"
            if bool(artifact["explicit_contrast"][target_index])
            else "observed_runner"
        ),
        "positive_id": int(artifact["positive_id"][target_index]),
        "negative_id": int(artifact["negative_id"][target_index]),
        "baseline_margin": baseline,
        "transition_seed_norm": transition_seed_norm,
        "transition_remote_effect": transition_effect,
        "transition_margin_per_seed_norm": effect_per_seed,
        "transition_nonadoption_score": -abs(effect_per_seed),
        "transition_zero_hop_effect": float(transition_hops[0]),
        "transition_one_hop_effect": float(transition_hops[1]),
        "transition_multi_hop_effect": float(transition_hops[2]),
        "rollout_multi_hop_negative": rollout_multi_hop_negative,
        "current_remote_write_effect": current_effect,
        "current_remote_zero_hop_effect": float(current_hops[0]),
        "current_remote_one_hop_effect": float(current_hops[1]),
        "current_remote_multi_hop_effect": float(current_hops[2]),
        "linearized_margin_without_current_remote_write": baseline - current_effect,
        "source_group_effect_l1": group_effect_l1,
        "source_group_effect_cancellation": cancellation,
        "semantic_source_roles_available": bool(artifact["semantic_source_roles_available"]),
        "closure_pass": closure_pass,
        "closure_margin_max_abs": float(artifact["closure_margin_max_abs"]),
        "closure_layer_max_abs": float(artifact["closure_layer_max_abs"]),
        "closure_root_max_abs": float(artifact["closure_root_max_abs"]),
        "binding_identified": False,
        "exact_intervention_run": False,
    }
    for index, name in enumerate(names):
        row[f"{name}_transition_coefficient"] = float(signed_coefficients[index])
        row[f"{name}_transition_coefficient_l1_share"] = float(l1_shares[index])
        row[f"{name}_seed_norm"] = float(seed_norms[index])
        row[f"{name}_transition_margin_effect"] = float(effects[index])
        row[f"{name}_layer_sign_reversal"] = bool(received_then_overridden(layer_effects[index]))
    return row


def source_unit_target_rows(
    artifact: dict[str, np.ndarray], target_index: int, *, identity: dict
) -> list[dict]:
    """Return one signed-effect row per annotated material source unit."""

    ids = np.asarray(artifact["source_unit_ids"], dtype=np.int64)
    roles = tuple(map(str, artifact["source_unit_roles"]))
    margins = np.asarray(artifact["source_unit_margin_response"], dtype=float)
    roots = np.asarray(artifact["source_unit_root_coefficient"], dtype=float)
    norms = np.asarray(artifact["source_unit_seed_norm"], dtype=float)
    if margins.shape[0] != len(ids) or roots.shape[0] != len(ids):
        raise ValueError("source-unit axes disagree")
    if len(roles) != len(ids) or norms.shape != (len(ids),):
        raise ValueError("source-unit metadata axes disagree")
    effects = margins[:, 0, :, target_index].sum(1)
    order = np.argsort(-np.abs(effects))
    ranks = np.empty(len(ids), dtype=int)
    ranks[order] = np.arange(1, len(ids) + 1)
    rows = []
    for index, unit_id in enumerate(ids):
        rows.append(
            {
                **identity,
                "source_unit_id": int(unit_id),
                "source_role": roles[index],
                "transition_coefficient": float(roots[index].sum()),
                "transition_coefficient_l1": float(np.abs(roots[index]).sum()),
                "seed_norm": float(norms[index]),
                "transition_margin_effect": float(effects[index]),
                "absolute_effect_rank": int(ranks[index]),
            }
        )
    return rows
