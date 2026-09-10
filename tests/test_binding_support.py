import json
import random
from collections import Counter

import pytest
import torch

from reanchor.binding_data import generate_dataset, load_examples, paired_examples
from reanchor.calibration import ReferenceCalibrator
from reanchor.features import candidate_ids
from reanchor.support import SupportHead, support_loss


def test_world_swap_keeps_all_asserted_history_true_and_fixed():
    examples = list(paired_examples("base", "train", random.Random(4)))
    assert len(examples) == 8
    for index in range(4):
        a, b = examples[index], examples[index + 4]
        pa, pb = a["program"], b["program"]
        assert pa["donor"] != pa["target"]
        assert Counter(pa["facts"].values()) == Counter(pb["facts"].values())
        for name in pa["asserted_entities"]:
            assert pa["facts"][name] == pb["facts"][name]
        a_start = a["supervision"]["binding_spans"][0][0]
        b_start = b["supervision"]["binding_spans"][0][0]
        assert a["response"][:a_start] == b["response"][:b_start]
        assert pa["answer"] != pb["answer"]
    # Explicit history changes must preserve target; referential changes must not.
    assert examples[0]["program"]["answer"] == examples[1]["program"]["answer"]
    assert examples[2]["program"]["answer"] != examples[3]["program"]["answer"]


def test_generated_splits_are_disjoint_and_output_not_overwritten(tmp_path):
    output = tmp_path / "dataset"
    manifest = generate_dataset(output, sources=8, seed=2)
    groups = [set(sources) for sources in manifest["assignments"].values()]
    assert len(set.union(*groups)) == sum(map(len, groups))
    assert all(load_examples(output / f"{split}.jsonl") for split in manifest["assignments"])
    with pytest.raises(FileExistsError):
        generate_dataset(output, sources=8)


def test_measurement_loader_rejects_hallucination_labels(tmp_path):
    example = next(paired_examples("base", "train", random.Random(4)))
    example["labels"] = []
    path = tmp_path / "leak.jsonl"
    path.write_text(json.dumps(example), encoding="utf-8")
    with pytest.raises(ValueError, match="labels forbidden"):
        load_examples(path)


def test_candidate_set_includes_source_outside_topk_and_observed():
    values = candidate_ids(torch.arange(20.0), [1, 2, 3], 4, [1], top_k=2)
    assert values == [2, 3, 4, 18, 19]


def test_support_has_no_source_free_value_and_chunking_is_equivalent():
    torch.manual_seed(4)
    head = SupportHead(12, width=8, blocks=2)
    source, address, candidates = torch.randn(9, 12), torch.randn(7, 12), torch.randn(11, 12)
    empty = head(source[:0], address, candidates)
    zeros = head(torch.zeros_like(source), address, candidates)
    assert torch.count_nonzero(empty) == 0
    assert torch.count_nonzero(zeros) == 0
    together = head(source, address, candidates, candidate_chunk=100)
    chunked = head(source, address, candidates, candidate_chunk=3)
    torch.testing.assert_close(together, chunked)
    loss = support_loss(together, "binding", 2) + support_loss(together, "neutral", 2)
    loss.backward()
    assert head.source_projection.weight.grad.abs().sum() > 0


def test_loss_is_direct_support_ce_not_native_prior():
    energies = torch.tensor([0.0, 2.0, -1.0], requires_grad=True)
    torch.testing.assert_close(support_loss(energies, "binding", 1), -energies.log_softmax(0)[1])
    assert support_loss(torch.ones(4), "neutral", 0).item() == 0


def reference_row(source, score, count=10, split="calibration"):
    return {
        "source_id": source,
        "split": split,
        "task": "QA",
        "raw_score": score,
        "negative_margin": score,
        "candidate_count": count,
        "source_length": 100,
    }


def test_reference_balances_sources_and_rejects_overlap():
    ref = [reference_row("long", 0)] * 100 + [reference_row("short", 2)]
    model = ReferenceCalibrator(ref, min_sources=2)
    row = model.transform(reference_row("test", 1, split="test"))
    assert row["score"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="disjoint"):
        model.transform(reference_row("long", 1))


def test_constant_reference_does_not_make_every_token_an_alarm():
    ref = [reference_row("a", 0), reference_row("b", 0)]
    row = ReferenceCalibrator(ref).transform(reference_row("new", 0, split="test"))
    assert row["score"] == 0
    assert row["calibration_level"] == 3  # small-pilot fallback explicitly reported
