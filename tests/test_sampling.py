import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from reanchor.cli import main


@pytest.fixture
def sample_input(tmp_path):
    torch.set_num_threads(1)
    tokenizer = Tokenizer(
        WordLevel({"[UNK]": 0, "facts": 1, ":": 2, "yes": 3, "no": 4}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    fast.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}:"
    model_path = tmp_path / "model"
    fast.save_pretrained(model_path)
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=5,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            eos_token_id=None,
            bos_token_id=None,
        )
    )
    model.save_pretrained(model_path)
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "source_info.jsonl").write_text(
        json.dumps({"source_id": "q1", "task_type": "QA", "prompt": "facts", "source_info": {}})
        + "\n",
        encoding="utf-8",
    )
    return model_path, dataset


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_sample_cli_saves_states_of_each_actual_generation_decision(sample_input, tmp_path, dtype):
    model_path, dataset = sample_input
    output = tmp_path / "samples"
    main(
        [
            "sample",
            "--dataset",
            str(dataset),
            "--source-ids",
            "q1",
            "--model",
            str(model_path),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            dtype,
            "--seeds",
            "2",
            "3",
            "--max-new-tokens",
            "3",
        ]
    )
    rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert [row["seed"] for row in rows] == [2, 3]
    model = LlamaForCausalLM.from_pretrained(
        model_path, attn_implementation="eager", torch_dtype=getattr(torch, dtype)
    ).eval()
    for row in rows:
        assert row["stop_reason"] == "max_new_tokens"
        with np.load(output / row["trace"]) as trace:
            assert trace["attention"].shape == (2, 2, 3, 5)
            assert trace["hidden_states"].shape == (3, 3, 16)
            ids = trace["token_ids"]
            prompt = int(trace["prompt_length"])
            for t in range(3):
                with torch.no_grad():
                    expected = model(
                        torch.tensor([ids[: prompt + t].tolist()]),
                        output_attentions=True,
                        output_hidden_states=True,
                    )
                np.testing.assert_allclose(
                    trace["top_logits"][t],
                    expected.logits[0, -1].float().topk(5).values.numpy(),
                    atol=3e-3 if dtype == "bfloat16" else 1e-5,
                )
                np.testing.assert_allclose(
                    trace["attention"][:, :, t, : prompt + t],
                    torch.stack([a[0, :, -1].float() for a in expected.attentions]),
                    atol=3e-3 if dtype == "bfloat16" else 5e-4,
                )
                np.testing.assert_allclose(
                    trace["hidden_states"][:, t],
                    torch.stack([h[0, -1].float() for h in expected.hidden_states]),
                    atol=2e-2 if dtype == "bfloat16" else 2e-3,
                )
                assert not trace["attention"][:, :, t, prompt + t :].any()
    assert not list(output.glob("*report*"))
    assert not list(output.rglob("trajectory.json"))


def test_sample_honors_model_stop_tokens(sample_input, tmp_path):
    model_path, dataset = sample_input
    config = json.loads((model_path / "generation_config.json").read_text())
    config["eos_token_id"] = [0, 1, 2, 3, 4]
    (model_path / "generation_config.json").write_text(json.dumps(config))
    output = tmp_path / "stopped"
    main(
        [
            "sample",
            "--dataset",
            str(dataset),
            "--source-ids",
            "q1",
            "--model",
            str(model_path),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--seeds",
            "1",
        ]
    )
    row = json.loads((output / "samples.jsonl").read_text())
    assert row["tokens"] == 1 and row["stop_reason"] == "eos"
