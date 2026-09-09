import numpy as np
import torch

from reanchor.capture.protocol import CapturePaths
from reanchor.tracing.cache import NativeCache
from reanchor.tracing.checkpoint import CheckpointWeights
from reanchor.tracing.jacobian import DifferentialLayer


def test_native_operator_loads_a_saved_llama_checkpoint_and_v3_shapes(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(4)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    checkpoint = tmp_path / "checkpoint"
    LlamaForCausalLM(config).save_pretrained(checkpoint)
    token_count, response_start = 6, 3
    rows = np.arange(response_start - 1, token_count)
    compact = tmp_path / "sample.npz"
    np.savez_compressed(
        compact,
        token_ids=np.arange(token_count),
        token_text=np.array([str(index) for index in range(token_count)]),
        row_position=rows,
        response_start=np.array(response_start),
        special_mask=np.zeros(token_count, dtype=bool),
        evidence_mask=np.zeros(token_count, dtype=bool),
    )
    qk = {}
    history = {}
    states = {"final_residual": np.random.randn(len(rows), 16).astype(np.float32)}
    for layer in range(2):
        qk.update(
            {
                f"query_{layer}": np.random.randn(4, len(rows), 4).astype(np.float32),
                f"key_{layer}": np.random.randn(2, token_count, 4).astype(np.float32),
                f"dtype_{layer}": np.array("float32"),
                f"scale_{layer}": np.array(0.5),
            }
        )
        history[f"L{layer}"] = np.zeros((4, len(rows), len(rows)), dtype=np.float32)
        states[f"residual_{layer}"] = np.random.randn(len(rows), 16).astype(np.float32)
        states[f"attention_{layer}"] = np.random.randn(len(rows), 16).astype(np.float32)
        states[f"value_{layer}"] = np.random.randn(2, token_count, 4).astype(np.float32)
    paths = CapturePaths(
        compact=compact,
        qk=compact.with_suffix(".qk.npz"),
        history=compact.with_suffix(".history.npz"),
        states=compact.with_suffix(".states.npz"),
        labels=compact.with_suffix(".labels.npz"),
        attention=compact.with_suffix(".attention.npz"),
    )
    np.savez_compressed(paths.qk, **qk)
    np.savez_compressed(paths.history, **history)
    np.savez_compressed(paths.states, **states)

    weights = CheckpointWeights(checkpoint)
    with NativeCache(paths, weights) as cache:
        operator = DifferentialLayer(cache, 0, chunk=2)
        same, cross, _ = operator.attention_jvp(torch.zeros(1, len(rows), config.hidden_size))

    assert same.shape == cross.shape == (1, len(rows), config.hidden_size)
