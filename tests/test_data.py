import json
import tempfile
import unittest
from pathlib import Path

from reanchor.data import load_events


class LoadEventsTest(unittest.TestCase):
    def test_loads_one_relation_event(self) -> None:
        record = {
            "schema": "reanchor/counterfactual-event@1",
            "event_id": "temporal-1",
            "source_id": "source-1",
            "split": "train",
            "relation": "temporal",
            "prompt_a": "The train leaves on Tuesday. What day does it leave?",
            "prompt_b": "The train leaves on Thursday. What day does it leave?",
            "answer_prefix": "The train leaves on ",
            "option_a": "Tuesday",
            "option_b": "Thursday",
            "followup": ". To confirm, it leaves on ",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            events = load_events(path)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_id, "temporal-1")
        self.assertEqual(events[0].source_id, "source-1")
        self.assertEqual(events[0].option_a, "Tuesday")

    def test_rejects_identical_counterfactual_worlds(self) -> None:
        record = {
            "schema": "reanchor/counterfactual-event@1",
            "event_id": "bad",
            "source_id": "source-bad",
            "split": "train",
            "relation": "temporal",
            "prompt_a": "same",
            "prompt_b": "same",
            "answer_prefix": "answer: ",
            "option_a": "Tuesday",
            "option_b": "Thursday",
            "followup": ". confirm: ",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "prompt_a and prompt_b"):
                load_events(path)

    def test_rejects_non_string_fields_at_the_input_boundary(self) -> None:
        record = {
            "schema": "reanchor/counterfactual-event@1",
            "event_id": 7,
            "source_id": "source-7",
            "split": "train",
            "relation": "temporal",
            "prompt_a": "world a",
            "prompt_b": "world b",
            "answer_prefix": "answer: ",
            "option_a": "Tuesday",
            "option_b": "Thursday",
            "followup": ". confirm: ",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "strings"):
                load_events(path)

    def test_rejects_undeclared_fields(self) -> None:
        record = {
            "schema": "reanchor/counterfactual-event@1",
            "event_id": "event-1",
            "source_id": "source-1",
            "split": "train",
            "relation": "temporal",
            "prompt_a": "world a",
            "prompt_b": "world b",
            "answer_prefix": "answer: ",
            "option_a": "Tuesday",
            "option_b": "Thursday",
            "followup": ". confirm: ",
            "label": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unexpected fields"):
                load_events(path)


if __name__ == "__main__":
    unittest.main()
