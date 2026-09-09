"""Validated explicit candidate contrasts shared by tracing and mechanism audit."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path


def read_contrasts(path_value: str | None) -> tuple[dict[str, list[dict]], str | None]:
    if path_value is None:
        return {}, None
    path = Path(path_value)
    payload = path.read_bytes()
    values = json.loads(payload)
    if not isinstance(values, dict):
        raise ValueError("contrast file must map sample keys to candidate lists")
    required = {"target", "positive_id", "negative_id"}
    for sample_key, entries in values.items():
        if not isinstance(sample_key, str) or not isinstance(entries, list):
            raise ValueError("contrast file must map sample keys to candidate lists")
        if any(not isinstance(entry, dict) or not required <= entry.keys() for entry in entries):
            raise ValueError("each contrast needs target, positive_id and negative_id")
    return values, sha256(payload).hexdigest()
