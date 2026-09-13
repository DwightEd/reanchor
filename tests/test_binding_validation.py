from collections import Counter

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from decoding.binding_validation import permutation_seed, summarize, validation_input


def test_all_layouts_keep_world_prefix_fixed_and_order_reversal_preserves_truth():
    backend = Tokenizer(WordLevel({"[UNK]": 0, "10": 1, "20": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{{ messages[0]['content'] }}\nAnswer:"
    case = dict(
        subjects=["Onions", "Potatoes"], predicate="require", values=["10", "20"], unit="minutes"
    )
    for subject in range(2):
        for layout in ("clean", "reverse", "distractor", "misbound"):
            worlds = [validation_input(tokenizer, case, w, subject, layout) for w in range(2)]
            assert worlds[0]["prefix"] == worlds[1]["prefix"]
            assert Counter(worlds[0]["ids"]) == Counter(worlds[1]["ids"])
            for world, item in enumerate(worlds):
                assert item["expected_token"] == 1 + (subject + world) % 2


def test_summary_keeps_control_coverage_and_pairs_at_template_level():
    rows = []
    for subject in range(2):
        for world in range(2):
            reading = {"valid": False}
            for method in ("context", "direct", "shuffled", "count"):
                reading[f"{method}_valid"] = method in {"direct", "count"}
                reading[f"{method}_support"] = [1, 0]
            rows.append(
                dict(
                    template="a",
                    layout="clean",
                    subject=subject,
                    world=world,
                    candidates=[10, 20],
                    expected_token=10,
                    model_correct=True,
                    readouts={"16": reading, "31": reading},
                )
            )
    report = summarize(rows)
    assert not report["error_detection_tested"]
    layer = report["layouts"]["clean"]["layers"]["16"]
    assert layer["common_valid"] == 0
    assert layer["methods"]["direct"]["valid"] == 4
    assert layer["methods"]["direct"]["correct"] == 4
    assert layer["methods"]["context"]["valid"] == 0
    assert len(layer["template_pairs"]) == 1
    assert layer["template_pairs"][0]["context_minus_direct"] == -1
    with pytest.raises(ValueError, match="both subjects"):
        summarize(rows + [rows[0]])
    rows[-1]["model_correct"] = False
    layer = summarize(rows)["layouts"]["clean"]["layers"]["16"]
    assert layer["actual_errors"] == 1
    assert layer["common_actual_errors"] == 0
    assert layer["methods"]["direct"]["valid_errors"] == 1
    assert layer["methods"]["context"]["valid_errors"] == 0
    assert layer["pairwise"]["context_vs_direct"]["conditions"] == 0


def test_endpoint_null_varies_across_conditions_but_is_resume_stable():
    identities = [
        (template, layout, subject, world, 16)
        for template in ("one", "two")
        for layout in ("clean", "reverse")
        for subject in range(2)
        for world in range(2)
    ]
    seeds = [permutation_seed(*identity) for identity in identities]
    assert len(set(seeds)) == len(seeds)
    assert seeds == [permutation_seed(*identity) for identity in identities]
    permutations = np.array([np.random.default_rng(seed).permutation(4) for seed in seeds])
    assert all(len(set(permutations[:, position])) > 1 for position in range(4))
