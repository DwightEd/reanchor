import json
from collections import Counter

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from decoding.mechanism import binding_input
from main import main


def test_binding_swap_changes_relation_without_changing_prefix_or_token_inventory():
    backend = Tokenizer(WordLevel({"[UNK]": 0, "10": 1, "20": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{{ messages[0]['content'] }}\nAnswer:"
    case = dict(
        subjects=["Onions", "Potatoes"], predicate="require", values=["10", "20"], unit="minutes"
    )
    first = binding_input(tokenizer, case, world=0, subject=0)
    swapped = binding_input(tokenizer, case, world=1, subject=0)
    assert first["expected"] == 0 and swapped["expected"] == 1
    assert Counter(first["ids"]) == Counter(swapped["ids"])
    p = first["prompt_length"]
    assert p == swapped["prompt_length"]
    assert first["ids"][p:] == swapped["ids"][p:]
    assert first["source"].any()
    assert set(first["spans"][0]).isdisjoint(first["spans"][1])
    other = binding_input(tokenizer, case, world=0, subject=1)
    assert other["expected"] == 1


def test_mechanism_cli_completes_all_controls_and_preserves_completed_cases(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "10": 1, "20": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{{ messages[0]['content'] }}\nAnswer:"
    model_path = tmp_path / "model"
    tokenizer.save_pretrained(model_path)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=3,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=256,
        )
    )
    model.save_pretrained(model_path)
    case = dict(
        id="binding",
        subjects=["Onions", "Potatoes"],
        predicate="require",
        values=["10", "20"],
        unit="minutes",
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(case) + "\n")
    output = tmp_path / "output"
    args = [
        "mechanism",
        "--model",
        str(model_path),
        "--cases",
        str(cases),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    main(args)
    complete = json.loads((output / "binding/complete.json").read_text())
    assert len(complete["checks"]) == 4
    assert all(r["sham_max_logit_error"] == 0 for r in complete["checks"])
    path = output / "binding/tokens.csv"
    before = path.stat().st_mtime_ns
    main(args)
    assert path.stat().st_mtime_ns == before
