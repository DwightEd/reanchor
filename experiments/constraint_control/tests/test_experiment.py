import json

import numpy as np

from experiments.constraint_control.config import ExperimentConfig
from experiments.constraint_control.experiment import ConstraintControlExperiment
from experiments.constraint_control.generation import CapturedGeneration, SamplingConfig


class RecordingBackend:
    def __init__(self):
        self.seeds = []

    @property
    def metadata(self):
        return {"model": "recording-model", "revision": "fixture"}

    def sample_and_replay(self, messages, sampling):
        self.seeds.append(sampling.seed)
        return CapturedGeneration(
            token_ids=np.array([1, 2, 3], dtype=np.int64),
            token_text=("prompt", " answer", " end"),
            special_mask=np.array([False, False, True]),
            response_start=1,
            response_text="answer end",
            generation_selected_logits=np.array([3.0, 4.0], dtype=np.float32),
            replay_selected_logits=np.array([3.0, 4.0], dtype=np.float32),
            model_logprobs=np.array([-0.2, -0.1], dtype=np.float32),
            sampling_logprobs=np.array([-0.2, -0.1], dtype=np.float32),
            entropy=np.array([0.5, 0.4], dtype=np.float32),
            top_token_ids=np.array([[2], [3]], dtype=np.int64),
            top_logits=np.array([[3.0], [4.0]], dtype=np.float32),
            residual_states=np.zeros((2, 2, 3), dtype=np.float16),
            attention_weights=np.zeros((1, 1, 2, 3), dtype=np.float16),
            replay_max_abs_logit_error=0.0,
            stop_reason="eos",
        )


def test_experiment_captures_each_source_and_seed_then_publishes_manifest(tmp_path):
    source = tmp_path / "questions.jsonl"
    records = [
        {
            "sample_id": sample_id,
            "source_id": f"source-{sample_id}",
            "split": "discovery",
            "task": "QA",
            "messages": [{"role": "user", "content": f"Question {sample_id}"}],
        }
        for sample_id in ("q1", "q2")
    ]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output = tmp_path / "run"
    backend = RecordingBackend()
    progress = []
    config = ExperimentConfig(
        input_path=source,
        output=output,
        model="recording-model",
        samplings=(
            SamplingConfig(seed=0, max_new_tokens=2, trace_top_k=1),
            SamplingConfig(seed=2, max_new_tokens=2, trace_top_k=1),
        ),
    )

    summary = ConstraintControlExperiment(
        config,
        backend=backend,
        progress=lambda completed, total, sample: progress.append(
            (completed, total, sample)
        ),
    ).run()

    assert summary == {"planned": 4, "completed": 4, "rejected": 0}
    assert backend.seeds == [0, 2, 0, 2]
    assert [(completed, total) for completed, total, _ in progress] == [
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    assert progress[-1][2] == "discovery/QA/q2 seed=2"
    manifest = json.loads((output / "index.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "constraint_control_run_v2"
    assert manifest["replay_mode"] == "same_prefix"
    assert manifest["labels_used_for_capture"] is False
    assert [sample["status"] for sample in manifest["samples"]] == ["completed"] * 4
    assert all((output / sample["trajectory"]).is_file() for sample in manifest["samples"])
