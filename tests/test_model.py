import math
import unittest
from types import SimpleNamespace

from reanchor.model import CausalLMScorer, encode_scoring_pair

try:
    import torch
except ImportError:
    torch = None


class FakeBatch(dict):
    def to(self, device):
        return FakeBatch({name: value.to(device) for name, value in self.items()})


class CharacterTokenizer:
    bos_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [{"a": 1, "b": 2, "c": 3}[character] for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def pad(self, values, *, padding: bool, return_tensors: str):
        assert padding and return_tensors == "pt"
        width = max(len(ids) for ids in values["input_ids"])
        padded = [ids + [9] * (width - len(ids)) for ids in values["input_ids"]]
        masks = [[1] * len(ids) + [0] * (width - len(ids)) for ids in values["input_ids"]]
        return FakeBatch(
            {
                "input_ids": torch.tensor(padded),
                "attention_mask": torch.tensor(masks),
            }
        )


class NextTokenModel:
    def __call__(self, *, input_ids, attention_mask, use_cache: bool):
        del attention_mask
        assert use_cache is False
        logits = torch.zeros((*input_ids.shape, 10))
        logits[:, :-1].scatter_(2, input_ids[:, 1:].unsqueeze(-1), 2.0)
        return SimpleNamespace(logits=logits)


class BoundaryMergingTokenizer:
    bos_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert text == "a b"
        assert not add_special_tokens and return_offsets_mapping
        return {"input_ids": [1, 2], "offset_mapping": [(0, 1), (1, 3)]}


class EncodeScoringPairTest(unittest.TestCase):
    def test_includes_a_token_that_merges_the_boundary_space(self) -> None:
        sequence, start, targets = encode_scoring_pair(BoundaryMergingTokenizer(), "a ", "b")

        self.assertEqual(sequence, [0, 1, 2])
        self.assertEqual(start, 2)
        self.assertEqual(targets, [2])


@unittest.skipIf(torch is None, "torch is not installed in the lightweight local test runtime")
class CausalLMScorerTest(unittest.TestCase):
    def test_scores_shifted_multi_token_continuations_with_right_padding(self) -> None:
        scorer = object.__new__(CausalLMScorer)
        scorer.torch = torch
        scorer.device = torch.device("cpu")
        scorer.tokenizer = CharacterTokenizer()
        scorer.model = NextTokenModel()

        scores = scorer.score([("a", "bc"), ("ab", "c")])

        expected = 2.0 - math.log(math.exp(2.0) + 9.0)
        self.assertAlmostEqual(scores[0], expected, places=6)
        self.assertAlmostEqual(scores[1], expected, places=6)


if __name__ == "__main__":
    unittest.main()
