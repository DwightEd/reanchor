from pathlib import Path

from experiments.constraint_control.main import parser


def test_command_line_exposes_explicit_sampling_and_capture_parameters():
    arguments = parser().parse_args(
        [
            "--input",
            "/data/RAGTruth/dataset",
            "--input-format",
            "ragtruth",
            "--task",
            "QA",
            "--split",
            "train",
            "--output",
            "runs/p001",
            "--model",
            "model-name",
            "--seeds",
            "1",
            "3",
            "--max-new-tokens",
            "32",
        ]
    )

    assert arguments.input == Path("/data/RAGTruth/dataset")
    assert arguments.input_format == "ragtruth"
    assert arguments.task == "QA"
    assert arguments.split == "train"
    assert arguments.output == Path("runs/p001")
    assert arguments.model == "model-name"
    assert arguments.seeds == [1, 3]
    assert arguments.max_new_tokens == 32
    assert arguments.replay_atol == 0.05
