"""Pure input/protocol tests; no model or network is needed."""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

module_path = Path(__file__).with_name("missing_constraint.py")
if not module_path.exists():
    module_path = Path(__file__).resolve().parents[1] / "src/decoding/missing_constraint.py"
spec = importlib.util.spec_from_file_location("missing_constraint", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    return "HEAD passage 1:" + module.OLD_SOURCE + " REST", (
        "First grill. Remove the bratwurst from the grill and cook the onions "
        "in the beer mixture for 10 to 12 minutes. NEXT"
    )


def test_action_surface_does_not_reuse_prior_steps_numbers():
    text = "4. Grill for 10 to 14 minutes. 5. Remove the bratwurst from the grill and serve."
    scoped = module.current_action_text(text)
    assert scoped == "Remove the bratwurst from the grill and serve."
    assert not module.surface_flags(scoped, truncated=False)["numeric_duration_mention"]


def test_exact_original_and_nine_worlds():
    prompt, response = sample()
    worlds = module.build_text_worlds(prompt, response)
    assert len(worlds) == 9
    assert worlds["original"]["prompt"] == prompt
    assert all(item["response"] == response for item in worlds.values())
    assert all(item["prompt"].startswith("HEAD passage 1:") for item in worlds.values())
    assert all(item["prompt"].endswith(" REST") for item in worlds.values())


def test_applicability_is_not_numeric_presence():
    worlds = module.build_text_worlds(*sample())
    assert worlds["before_only"]["applicable_duration"] is None
    assert "12 minutes" in worlds["before_only"]["prompt"]
    assert worlds["wrong_object"]["applicable_duration"] is None
    assert "14 minutes" in worlds["wrong_object"]["prompt"]
    assert worlds["after_only"]["applicable_duration"] == [10, 14]


def test_relation_only_applicability_control():
    worlds = module.build_text_worlds(*sample())
    before = worlds["before_only"]["prompt"]
    after = worlds["after12_aligned"]["prompt"]
    assert Counter(before.split()) == Counter(after.split())
    assert worlds["after12_aligned"]["applicable_duration"] == [10, 12]
    assert before.replace("Before", "STAGE").replace("After", "Before").replace("STAGE", "After") == after


def test_missing_and_explicit_unspecified_are_separate():
    worlds = module.build_text_worlds(*sample())
    assert "unspecified" not in worlds["neither"]["prompt"]
    assert "unspecified" in worlds["explicit_unspecified"]["prompt"]
    assert worlds["explicit_unspecified"]["pool_with_absence"] is False


def test_prefixes_stop_before_commitment():
    _, response = sample()
    prefixes = module.build_prefixes(response)
    assert len(prefixes) == 3
    assert prefixes["action"]["text"].endswith("from the grill and")
    assert prefixes["uncommitted"]["text"].endswith("in the beer mixture")
    assert " for " not in prefixes["uncommitted"]["text"]
    assert prefixes["committed"]["text"].endswith("for 10 to ")
    assert all("NEXT" not in x["text"] for x in prefixes.values())


def test_ambiguous_sources_are_rejected():
    prompt, response = sample()
    with pytest.raises(ValueError, match="source"):
        module.build_text_worlds(prompt + module.OLD_SOURCE, response)


def test_ambiguous_actions_are_rejected():
    prompt, response = sample()
    with pytest.raises(ValueError, match="action"):
        module.build_text_worlds(prompt, response + response)


def test_surface_annotation_is_not_ground_truth():
    result = module.surface_flags(" for 10 to 12 minutes.", truncated=False)
    assert result["numeric_duration_mention"] is True
    assert result["hallucination_label"] is None
    result = module.surface_flags(" Stir the onions.", truncated=True)
    assert result["absence_of_duration_confirmed"] is False
    assert result["requires_semantic_review"] is True


def test_graph_memory_estimate_is_monotone():
    assert module.graph_bytes(100, 32, 32, 4096, 8, 128) > 0
    assert module.graph_bytes(101, 32, 32, 4096, 8, 128) > module.graph_bytes(
        100, 32, 32, 4096, 8, 128
    )
