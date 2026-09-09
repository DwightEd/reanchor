"""Narrow, read-only adapter from one v3 capture sample to tracing arrays."""

from __future__ import annotations

import re
from contextlib import ExitStack

import numpy as np

from reanchor.capture.protocol import CapturePaths

CONTROL = re.compile(
    r"<\|(?:begin_of_text|end_of_text|start_header_id|end_header_id|eot_id|eom_id|"
    r"finetune_right_pad_id|python_tag|reserved_special_token_\d+)\|>"
)
META_FIELDS = (
    "token_ids",
    "token_text",
    "special_mask",
    "source_unit_id",
    "evidence_mask",
    "row_position",
    "response_start",
    "predictor_logprob",
    "generator_model",
    "settings",
    "capture_special_mask",
    "readout_runner_id",
)


def read_trace(path) -> dict[str, np.ndarray]:
    """Read metadata only; large feature arrays remain unopened."""

    with np.load(path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in META_FIELDS if key in archive}
    native = np.asarray(trace.get("capture_special_mask", trace["special_mask"]), bool)
    token_text = trace.get("token_text", np.full(len(native), ""))
    controls = np.array([bool(CONTROL.fullmatch(str(token))) for token in token_text])
    trace["capture_special_mask"] = native.copy()
    trace["special_mask"] = np.asarray(trace["special_mask"], bool) | controls
    return trace


class NativeCache:
    """Keep one sample's memory-mapped native states open during tracing."""

    def __init__(self, paths: CapturePaths, weights):
        self.paths = paths
        self.weights = weights
        self.stack = ExitStack()
        self.event_readouts = {}
        self.trace = read_trace(paths.compact)
        self.states = self.stack.enter_context(np.load(paths.states, allow_pickle=False))
        self.history = self.stack.enter_context(np.load(paths.history, allow_pickle=False))
        self.qk = self.stack.enter_context(np.load(paths.qk, allow_pickle=False))
        config = weights.config
        self.layers = int(config["num_hidden_layers"])
        self.heads = int(config["num_attention_heads"])
        self.rows = len(self.trace["row_position"])
        query = self.qk["query_0"]
        if query.shape[:2] != (self.heads, self.rows):
            self.stack.close()
            raise ValueError("capture and checkpoint disagree on attention heads or rows")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.event_readouts.clear()
        return self.stack.__exit__(*args)
