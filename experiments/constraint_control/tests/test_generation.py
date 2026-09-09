import json

import numpy as np

from experiments.constraint_control.generation import (
    CapturedGeneration,
    GenerationRecorder,
    SamplingConfig,
)
from experiments.constraint_control.records import ChatMessage, SourceRecord


class DeterministicBackend:
    @property
    def metadata(self):
        return {"model": "deterministic-test-model", "revision": "fixture"}

    def sample_and_replay(self, messages, sampling):
        assert messages == (ChatMessage(role="user", content="Use fact A."),)
        assert sampling.seed == 7
        return CapturedGeneration(
            token_ids=np.array([10, 11, 12, 13], dtype=np.int64),
            token_text=("Use", " fact", " A", "."),
            response_start=2,
            response_text=" A.",
            generation_selected_logits=np.array([4.0, 5.0], dtype=np.float32),
            replay_selected_logits=np.array([4.0, 5.0], dtype=np.float32),
            model_logprobs=np.array([-0.2, -0.1], dtype=np.float32),
            sampling_logprobs=np.array([-0.3, -0.2], dtype=np.float32),
            entropy=np.array([0.8, 0.7], dtype=np.float32),
            top_token_ids=np.array([[12, 9], [13, 8]], dtype=np.int64),
            top_logits=np.array([[4.0, 3.0], [5.0, 2.0]], dtype=np.float32),
            residual_states=np.zeros((3, 2, 4), dtype=np.float16),
            attention_weights=np.zeros((2, 1, 2, 4), dtype=np.float16),
            replay_max_abs_logit_error=0.0,
            stop_reason="eos",
        )


def test_recorder_persists_a_label_free_free_run_trajectory(tmp_path):
    record = SourceRecord(
        sample_id="question/1",
        source_id="source-a",
        split="discovery",
        task="QA",
        messages=(ChatMessage(role="user", content="Use fact A."),),
    )
    recorder = GenerationRecorder(DeterministicBackend(), tmp_path, replay_atol=1e-4)

    artifact = recorder.run(record, SamplingConfig(seed=7, max_new_tokens=2, trace_top_k=2))

    assert artifact.sample_key == "discovery/QA/question/1"
    assert artifact.response_tokens == 2
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema"] == "constraint_control_trajectory_v1"
    assert metadata["labels_used_for_capture"] is False
    assert metadata["sampling"]["seed"] == 7
    assert metadata["model"]["model"] == "deterministic-test-model"
    with np.load(artifact.capture_path, allow_pickle=False) as capture:
        np.testing.assert_array_equal(capture["token_ids"], [10, 11, 12, 13])
        np.testing.assert_array_equal(capture["row_position"], [1, 2])
        assert capture["residual_states"].shape == (3, 2, 4)
        assert capture["attention_weights"].shape == (2, 1, 2, 4)
