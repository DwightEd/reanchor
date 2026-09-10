"""Build post-selection token and future-outcome records from frozen artifacts."""

from __future__ import annotations

import numpy as np


def load_labels(dataset, sample) -> np.ndarray | None:
    path = dataset.paths(sample).labels
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        labels = np.asarray(archive["labels"], dtype=np.int8)
    if len(labels) != sample.response_tokens or not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError(f"{sample.key}: labels must cover response tokens with -1/0/1")
    return labels


def target_phase(labels: np.ndarray, response_index: int) -> str:
    """Name a known target's position relative to a hallucinated span."""

    if int(labels[response_index]) == 0:
        return "normal"
    if response_index == 0 or int(labels[response_index - 1]) not in (0, 1):
        return "hallucinated_boundary_unknown"
    return "onset" if int(labels[response_index - 1]) == 0 else "continuing"


def join_outcomes(rows: list[dict], labels: np.ndarray | None) -> list[dict]:
    """Attach N/H outcomes after label-free signal construction."""

    labels = None if labels is None else np.asarray(labels, dtype=np.int8)
    joined = []
    for source in rows:
        row = dict(source)
        index = int(row["target_response_index"])
        label = int(labels[index]) if labels is not None and 0 <= index < len(labels) else -1
        row["label"] = label if label in (0, 1) else ""
        row["phase"] = target_phase(labels, index) if label in (0, 1) else ""
        joined.append(row)
    return joined


def event_records(sample, transitions, events, labels) -> list[dict]:
    positions = np.asarray(events["row_position"])
    eligible = np.asarray(events["eligible"], dtype=bool)
    response_start = int(transitions["response_start"])
    token_count = len(transitions.get("special_mask", []))
    special = np.asarray(
        transitions.get("special_mask", np.zeros(max(token_count, positions.max() + 2), bool))
    )
    rows = []
    for index in np.flatnonzero(eligible):
        target = int(positions[index]) + 1
        label_index = target - response_start
        label = ""
        if (
            labels is not None
            and 0 <= label_index < len(labels)
            and (target >= len(special) or not special[target])
            and int(labels[label_index]) in (0, 1)
        ):
            label = int(labels[label_index])
        rows.append(
            {
                "split": sample.split,
                "task": sample.task,
                "sample_id": sample.sample_id,
                "source_id": sample.source_id,
                "query_position": int(positions[index]),
                "target_position": target,
                "label": label,
                "significant": bool(events["significant"][index]),
                "anchor": bool(events["anchor"][index]),
                "episode_id": int(events["episode_id"][index]),
                "channel": str(events["channel"][index]),
                "reanchor_type": str(events["reanchor_type"][index]),
                "family_p_value": float(events["family_p_value"][index]),
                "sparse_score": float(events["sparse_score"][index]),
                "broad_score": float(events["broad_score"][index]),
            }
        )
    return rows


def future_records(sample, transitions, events, labels, horizons) -> list[dict]:
    if labels is None:
        return []
    response_start = int(transitions["response_start"])
    positions = np.asarray(events["row_position"])
    special = np.asarray(
        transitions.get("special_mask", np.zeros(response_start + len(labels), bool))
    )
    rows = []
    for anchor_index in np.flatnonzero(events["anchor"]):
        first_label = int(positions[anchor_index]) + 1 - response_start
        kind = str(events["reanchor_type"][anchor_index])
        for lower, upper in horizons:
            horizon = f"{lower}-{upper}"
            for offset in range(lower, upper + 1):
                label_index = first_label + offset
                token_position = response_start + label_index
                if not 0 <= label_index < len(labels):
                    continue
                if token_position < len(special) and special[token_position]:
                    continue
                label = int(labels[label_index])
                if label in (0, 1):
                    rows.append(
                        {
                            "source_id": sample.source_id,
                            "group": f"{sample.split}/{sample.task}",
                            "reanchor_type": kind,
                            "horizon": horizon,
                            "hallucinated": label == 1,
                        }
                    )
    return rows
