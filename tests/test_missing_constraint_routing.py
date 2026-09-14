import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).with_name("missing_constraint_routing.py")
if not path.exists():
    path = Path(__file__).resolve().parents[1] / "src/decoding/missing_constraint_routing.py"
spec = importlib.util.spec_from_file_location("missing_constraint_routing", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_query_partition_predicts_next_token_without_overlap():
    groups = module.query_groups(5, 9)
    assert groups == {"last": [8], "earlier": [4, 5, 6, 7], "all": [4, 5, 6, 7, 8]}
    assert set(groups["last"]).isdisjoint(groups["earlier"])


def test_query_partition_rejects_missing_response():
    with pytest.raises(ValueError):
        module.query_groups(5, 5)


def test_pair_is_only_two_prompt_relation_tokens():
    a = {"input_ids": [1, 2, 3, 4, 5, 6], "prompt_length": 4, "source_mask": [False, True, True, False]}
    b = {**a, "input_ids": [1, 3, 2, 4, 5, 6]}
    assert module.validate_pair(a, b) == [1, 2]
    with pytest.raises(ValueError):
        module.validate_pair(a, {**b, "input_ids": [1, 3, 2, 4, 6, 5]})


def test_pair_rejects_changed_source_partition():
    a = {"input_ids": [1, 2, 3, 4, 5, 6], "prompt_length": 4, "source_mask": [False, True, True, False]}
    b = {**a, "input_ids": [1, 3, 2, 4, 5, 6], "source_mask": [True, True, True, False]}
    with pytest.raises(ValueError):
        module.validate_pair(a, b)


def test_relation_swap_checks_decoded_meaning_not_only_bag():
    a = {"input_ids": [1, 2, 3, 4, 5, 6], "prompt_length": 4, "source_mask": [False, True, True, False]}
    b = {**a, "input_ids": [1, 3, 2, 4, 5, 6]}
    assert module.validate_relation_tokens(a, b, lambda ids: {2: " Before", 3: " After"}[ids[0]]) == [1, 2]
    with pytest.raises(ValueError):
        module.validate_relation_tokens(a, b, lambda ids: str(ids[0]))
