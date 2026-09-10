import json
import tempfile
import unittest
from pathlib import Path

from reanchor.experiment import ConstraintControlExperiment, ExperimentConfig


class FakeScorer:
    metadata = {"model": "fake", "revision": "test"}

    def render_prompt(self, prompt: str) -> str:
        return prompt

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for context, continuation in pairs:
            world_a = "Tuesday" in context.split("What day", maxsplit=1)[0]
            committed_a = "answer: Tuesday" in context
            committed_b = "answer: Thursday" in context
            if not committed_a and not committed_b:
                prefers_a = world_a
            else:
                prefers_a = committed_a
            scores.append(1.0 if (continuation == "Tuesday") == prefers_a else -1.0)
        return scores


class ConstraintControlExperimentTest(unittest.TestCase):
    def test_writes_one_compact_causal_record(self) -> None:
        event = {
            "schema": "reanchor/counterfactual-event@1",
            "event_id": "temporal-1",
            "source_id": "source-1",
            "split": "train",
            "relation": "temporal",
            "prompt_a": "Tuesday fact. What day?",
            "prompt_b": "Thursday fact. What day?",
            "answer_prefix": "answer: ",
            "option_a": "Tuesday",
            "option_b": "Thursday",
            "followup": ". confirm: ",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "events.jsonl"
            output_path = root / "run"
            input_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            config = ExperimentConfig(
                input_path=input_path,
                output_path=output_path,
                model_path=Path("unused-in-test"),
                show_progress=False,
            )

            summary = ConstraintControlExperiment(config, scorer=FakeScorer()).run()
            result = json.loads((output_path / "events.jsonl").read_text(encoding="utf-8"))
            snapshot_exists = (output_path / "input.jsonl").is_file()

        self.assertEqual(summary["events"], 1)
        self.assertEqual(summary["model"], FakeScorer.metadata)
        self.assertEqual(len(summary["input"]["sha256"]), 64)
        self.assertTrue(snapshot_exists)
        self.assertEqual(result["schema"], "reanchor/constraint-control@2")
        self.assertEqual(result["event_id"], "temporal-1")
        self.assertEqual(result["source_id"], "source-1")
        self.assertEqual(result["split"], "train")
        self.assertEqual(result["logp"]["onset_a"], [1.0, -1.0])
        self.assertEqual(
            result["effects"],
            {
                "source_onset": 2.0,
                "source_followup": 0.0,
                "prefix_followup": 2.0,
                "source_prefix_coupling": 0.0,
            },
        )

    def test_refuses_to_overwrite_a_nonempty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "events.jsonl"
            input_path.write_text("{}\n", encoding="utf-8")
            output_path = root / "run"
            output_path.mkdir()
            sentinel = output_path / "summary.json"
            sentinel.write_text("keep", encoding="utf-8")
            config = ExperimentConfig(
                input_path=input_path,
                output_path=output_path,
                model_path=Path("unused-in-test"),
                show_progress=False,
            )

            with self.assertRaisesRegex(FileExistsError, "not empty"):
                ConstraintControlExperiment(config, scorer=FakeScorer()).run()

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
