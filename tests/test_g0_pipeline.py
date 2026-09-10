"""Real tiny random Llama integration. Outputs are software tests, NOT research evidence."""

import json
import random
import shutil

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from reanchor.binding_data import paired_examples
from reanchor.calibration import calibrate_scores
from reanchor.cli import parser
from reanchor.evaluation import evaluate_ragtruth
from reanchor.features import FeatureStore, FrozenBackbone, extract_features
from reanchor.io import write_jsonl
from reanchor.runner import run_pipeline
from reanchor.training import load_head


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    torch.set_num_threads(1)
    folder = tmp_path_factory.mktemp("tiny-local-llama")
    pre = Whitespace()
    words = set()
    for seed in (42, 5):
        for example in paired_examples("fixture", "test", random.Random(seed)):
            for field in ("source", "instruction", "response"):
                words.update(token for token, _ in pre.pre_tokenize_str(example[field]))
    words.update(
        "User Assistant Evidence Task Answer using Monday Tuesday Wednesday Thursday Friday "
        "Saturday Sunday Later Instead Summary day correct wrong".split()
    )
    vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3}
    vocab.update({word: index + 4 for index, word in enumerate(sorted(words))})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    fast.save_pretrained(folder)
    torch.manual_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=1024,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.save_pretrained(folder)
    return folder


def test_frozen_replay_has_no_future_suffix_leakage(tiny_checkpoint, tmp_path):
    base = {
        "schema": "reanchor/response@1",
        "source_id": "natural-fixture",
        "split": "test",
        "task": "QA",
        "instruction": "State the opening day.",
        "source": "Gallery opens on Monday.",
    }
    examples = [
        {**base, "id": "a", "response": "Gallery opens on Monday. Later Friday."},
        {**base, "id": "b", "response": "Gallery opens on Monday. Instead Sunday."},
    ]
    path = tmp_path / "input.jsonl"
    write_jsonl(path, examples)
    backbone = FrozenBackbone(tiny_checkpoint, "cpu", "float32", 1024)
    extract_features(path, tmp_path / "cache", backbone, progress=False)
    store = FeatureStore(tmp_path / "cache")
    a = next(r for r in store.samples if r["response_id"] == "a" and r["token_index"] == 3)
    b = next(r for r in store.samples if r["response_id"] == "b" and r["token_index"] == 3)
    assert a["candidate_ids"] == b["candidate_ids"]
    assert a["negative_margin"] == pytest.approx(b["negative_margin"], abs=1e-5)
    for first, second in zip(store.tensors(a, "cpu"), store.tensors(b, "cpu"), strict=True):
        torch.testing.assert_close(first, second, atol=1e-5, rtol=1e-5)
    assert a["address_end"] == len(backbone.prompt_ids(base["instruction"])) + 3
    with (tmp_path / "cache" / a["source_file"]).open("ab") as stream:
        stream.write(b"tampered-test-fixture")
    with pytest.raises(ValueError, match="tensor changed"):
        FeatureStore(tmp_path / "cache")


def test_backbone_fingerprint_binds_weight_contents(tiny_checkpoint, tmp_path):
    original = FrozenBackbone(tiny_checkpoint, "cpu", "float32", 1024)
    changed = tmp_path / "changed-weights"
    shutil.copytree(tiny_checkpoint, changed)
    with torch.no_grad():
        original.model.get_input_embeddings().weight[0, 0] += 1
    original.model.save_pretrained(changed)
    altered = FrozenBackbone(changed, "cpu", "float32", 1024)
    assert original.metadata["fingerprint"] != altered.metadata["fingerprint"]


def test_complete_cpu_pipeline_with_real_frozen_llama_and_label_join(tiny_checkpoint, tmp_path):
    dataset = tmp_path / "ragtruth-fixture"
    dataset.mkdir()
    write_jsonl(
        dataset / "source_info.jsonl",
        [
            {
                "source_id": str(i),
                "task_type": "QA",
                "source_info": {
                    "question": "State the opening day.",
                    "passages": "Gallery opens on Monday.",
                },
            }
            for i in range(4)
        ],
    )
    write_jsonl(
        dataset / "response.jsonl",
        [
            {
                "id": str(i),
                "source_id": str(i),
                "model": "software-test-fixture",
                "response": "Friday Monday.",
                "labels": [{"start": 0, "end": 6, "text": "Friday"}],
            }
            for i in range(4)
        ],
    )
    output = tmp_path / "run"
    args = parser().parse_args(
        [
            "pipeline",
            "--model",
            str(tiny_checkpoint),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--sources",
            "8",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--width",
            "8",
            "--max-tokens",
            "1024",
            "--ragtruth",
            str(dataset),
            "--bootstrap",
            "2",
            "--no-progress",
        ]
    )
    result = run_pipeline(args)
    assert result["natural_evaluation"] == "evaluation/summary.json"
    assert (output / "head/head.pt").is_file()
    evaluation = json.loads((output / "evaluation/summary.json").read_text())
    assert evaluation["metrics"]["first_error_full_stream"]["score"]["positive"] == 3
    assert evaluation["alarms"]["error_responses"] == 3
    prepared = (output / "natural/test.jsonl").read_text()
    assert '"labels"' not in prepared
    frozen = (output / "scores/natural/test/scores.jsonl").read_text()
    assert '"hallucinated"' not in frozen
    with pytest.raises(FileExistsError):
        run_pipeline(args)
    # Same checkpoint cannot be used with changed extraction settings.
    store = FeatureStore(output / "cache/program/test")
    store.manifest["extraction_identity"] = "different-top-k-or-dtype"
    with pytest.raises(ValueError, match="extraction identity"):
        load_head(output / "head/head.pt", store, "cpu")
    # Frozen calibration/test must also agree on their extraction provenance.
    ref_manifest = output / "scores/natural/calibration/manifest.json"
    meta = json.loads(ref_manifest.read_text())
    original_identity = meta["extraction_identity"]
    meta["extraction_identity"] = "different-extractor"
    ref_manifest.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="extraction identity"):
        calibrate_scores(
            output / "scores/natural/calibration",
            output / "scores/natural/test",
            tmp_path / "invalid-calibration",
        )
    meta["extraction_identity"] = original_identity
    meta["preparation_identity"] = "other-generator-or-split-policy-same-raw-files"
    ref_manifest.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="preparation split/configuration"):
        calibrate_scores(
            output / "scores/natural/calibration",
            output / "scores/natural/test",
            tmp_path / "mixed-preparation-calibration",
        )
    # A source-only dataset mutation cannot silently reuse frozen scores.
    sources_path = dataset / "source_info.jsonl"
    sources_path.write_text(sources_path.read_text().replace("Monday", "Sunday"), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset contents changed"):
        evaluate_ragtruth(output / "calibrated/natural", dataset, tmp_path / "invalid-evaluation")


def test_cli_disallows_output_abbreviation():
    with pytest.raises(SystemExit) as error:
        parser().parse_args(
            ["pipeline", "--model", "local", "--output", "intended", "--out", "redirected"]
        )
    assert error.value.code == 2
