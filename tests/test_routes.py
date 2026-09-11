import csv
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE, WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from main import main


@pytest.fixture
def route_input(tmp_path):
    prompt = (
        "Question:\npassage 1:A\n\npassage 2:B\n\npassage 3:C\n\n"
        'In case the passages do not contain the necessary information, reply "Unable".\noutput:'
    )
    response = "abcd"
    vocabulary = {"[UNK]": 0, "<s>": 1}
    vocabulary.update({char: i + 2 for i, char in enumerate(sorted(set(prompt + response)))})
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="<s>"
    )
    tokenizer.chat_template = "<s>{{ messages[0]['content'] }}:"
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True
    )
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    ids = np.array(prompt_ids + response_ids)
    weights = np.zeros((1, 2, 4, len(ids)), dtype=np.float32)
    a, b = prompt_ids.index(vocabulary["A"]), prompt_ids.index(vocabulary["B"])
    weights[0, 0, 0, a] = weights[0, 1, 0, b] = 0.6
    weights[0, :, 1, a] = weights[0, :, 1, b] = 0.3
    weights[0, :, 2, a] = weights[0, :, 2, b] = 0.15
    weights[0, 0, 3, b] = weights[0, 1, 3, a] = 0.3
    weights[0, :, :, 0] = [[0.4, 0.4, 0.7, 0.7]] * 2
    np.savez(
        tmp_path / "00012.npz",
        attention=weights,
        token_ids=ids,
        prompt_length=len(prompt_ids),
        special_mask=np.isin(ids, tokenizer.all_special_ids),
        token_pieces=tokenizer.convert_ids_to_tokens(ids.tolist()),
        token_text=[tokenizer.decode([int(i)]) for i in ids],
        top_logits=np.tile([3.0, 2.0], (4, 1)),
    )
    (tmp_path / "samples.jsonl").write_text(
        json.dumps({"source_id": "14375", "seed": 0, "response": response, "trace": "00012.npz"})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "prompts.jsonl").write_text(
        json.dumps({"source_id": "14375", "prompt": prompt}) + "\n", encoding="utf-8"
    )
    (tmp_path / "settings.json").write_text(
        json.dumps({"model": str(tmp_path / "tokenizer")}), encoding="utf-8"
    )
    (tmp_path / "cases.csv").write_text("source_id,seed,focus\n14375,0,c\n", encoding="utf-8")
    return tmp_path


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_routes_distinguish_disagreement_from_diffusion_and_ignore_sink_mass(route_input):
    output = route_input / "analysis"
    main(
        [
            "routes",
            "--samples",
            str(route_input),
            "--cases",
            str(route_input / "cases.csv"),
            "--output",
            str(output),
            "--before",
            "1",
            "--after",
            "1",
        ]
    )
    result = output / "00012"
    rows = read_csv(result / "trajectory.csv")
    assert [float(row["dispersion"]) for row in rows] == pytest.approx([0, 1, 1, 0])
    assert [float(row["disagreement"]) for row in rows] == pytest.approx([1, 0, 0, 1])
    assert [float(row["source_mass"]) for row in rows] == pytest.approx([0.6, 0.6, 0.3, 0.3])
    assert rows[0]["shift"] == rows[0]["shift_baseline"] == ""
    assert [float(row["shift"]) for row in rows[1:]] == pytest.approx([0.5, 0, 0.5])
    assert float(rows[2]["shift_baseline"]) == 0.5
    tokens = read_csv(result / "tokens.csv")
    assert [row["token"] for row in tokens] == list("abcd")
    assert [row["focus"] for row in tokens] == ["0", "0", "1", "0"]
    assert all(row["logit_entropy"] == "" for row in tokens)
    assert [row["token"] for row in read_csv(result / "source_tokens.csv")] == list("ABC")
    reading = read_csv(result / "reading.csv")
    assert sorted({int(row["step"]) for row in reading}) == [1, 2, 3]
    assert reading[0]["gain_context"] == "B"
    assert float(reading[0]["gain"]) == pytest.approx(0.3)
    assert float(reading[2]["shift"]) == pytest.approx(0)
    assert int(tokens[2]["query_position"]) + 1 == int(tokens[2]["position"])
    assert (result / "trajectory.png").stat().st_size > 1000


def test_routes_do_not_invent_reading_on_inactive_edges_or_zero_source_heads(route_input):
    with np.load(route_input / "00012.npz") as trace:
        arrays = dict(trace)
    arrays["attention"][:, :, 3] = 0
    arrays["attention"][:, :, 3, 0] = 1
    np.savez(route_input / "00012.npz", **arrays)
    output = route_input / "zero"
    main(
        [
            "routes",
            "--samples",
            str(route_input),
            "--cases",
            str(route_input / "cases.csv"),
            "--output",
            str(output),
        ]
    )
    rows = read_csv(output / "00012/trajectory.csv")
    assert rows[3]["dispersion"] == rows[3]["disagreement"] == rows[3]["shift"] == ""
    assert int(rows[3]["valid_heads"]) == 0
    reading = read_csv(output / "00012/reading.csv")
    loss = next(row for row in reading if row["step"] == "2" and row["head"] == "0")
    assert float(loss["gain"]) == pytest.approx(-0.15)
    assert loss["gain_context"] in {"A", "B"}
    empty = next(row for row in reading if row["step"] == "3")
    assert empty["peak_context"] == empty["dispersion"] == empty["shift"] == ""


def test_focus_includes_all_generated_byte_tokens_of_a_unicode_character(route_input):
    prompt = json.loads((route_input / "prompts.jsonl").read_text())["prompt"]
    vocabulary = {"[UNK]": 0, "<s>": 1}
    vocabulary.update(
        {char: i + 2 for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    )
    backend = Tokenizer(BPE(vocabulary, [], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="<s>"
    )
    tokenizer.chat_template = "<s>{{ messages[0]['content'] }}:"
    tokenizer.save_pretrained(route_input / "tokenizer")
    prompt_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}])
    response_ids = tokenizer.encode("aébc", add_special_tokens=False)
    assert len(response_ids) == 5
    ids = np.array(prompt_ids + response_ids)
    attention = np.zeros((1, 2, 5, len(ids)), dtype=np.float32)
    attention[:, :, :, prompt_ids.index(vocabulary["A"])] = 1
    np.savez(
        route_input / "00012.npz",
        attention=attention,
        token_ids=ids,
        prompt_length=len(prompt_ids),
        token_pieces=tokenizer.convert_ids_to_tokens(ids.tolist()),
        token_text=[tokenizer.decode([int(i)]) for i in ids],
        top_logits=np.tile([3, 2], (5, 1)),
    )
    sample = json.loads((route_input / "samples.jsonl").read_text())
    sample["response"] = "aébc"
    (route_input / "samples.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    (route_input / "cases.csv").write_text("source_id,seed,focus\n14375,0,é\n", encoding="utf-8")
    output = route_input / "unicode"
    main(
        [
            "routes",
            "--samples",
            str(route_input),
            "--cases",
            str(route_input / "cases.csv"),
            "--output",
            str(output),
        ]
    )
    rows = read_csv(output / "00012/tokens.csv")
    assert [int(row["step"]) for row in rows if row["focus"] == "1"] == [1, 2]


def test_route_curves_and_prior_baseline_do_not_use_future_decisions(route_input):
    command = ["routes", "--samples", str(route_input), "--cases", str(route_input / "cases.csv")]
    main(command + ["--output", str(route_input / "original")])
    with np.load(route_input / "00012.npz") as trace:
        arrays = dict(trace)
    arrays["attention"][:, :, 3, :] = 0
    arrays["attention"][:, :, 3, 0] = 1
    np.savez(route_input / "00012.npz", **arrays)
    main(command + ["--output", str(route_input / "future")])
    for filename in ("trajectory.csv", "reading.csv"):
        original = read_csv(route_input / "original/00012" / filename)
        changed = read_csv(route_input / "future/00012" / filename)
        assert [r for r in original if int(r["step"]) < 3] == [
            r for r in changed if int(r["step"]) < 3
        ]


@pytest.mark.parametrize("problem", ["prompt", "quote", "duplicate"])
def test_routes_reject_misaligned_inputs_instead_of_guessing(route_input, problem):
    if problem == "prompt":
        path = route_input / "prompts.jsonl"
        path.write_text(path.read_text().replace("passage 1:A", "passage 1:B"))
        expected = "raw prompt"
    elif problem == "quote":
        (route_input / "cases.csv").write_text("source_id,seed,focus\n14375,0,missing\n")
        expected = "focus quote"
    else:
        path = route_input / "samples.jsonl"
        path.write_text(path.read_text() * 2)
        expected = "one captured sample"
    with pytest.raises(ValueError, match=expected):
        main(
            [
                "routes",
                "--samples",
                str(route_input),
                "--cases",
                str(route_input / "cases.csv"),
                "--output",
                str(route_input / "invalid"),
            ]
        )


def test_routes_use_captured_prefix_even_if_the_current_chat_template_changed(route_input):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(route_input / "tokenizer")
    tokenizer.chat_template += ":"
    tokenizer.save_pretrained(route_input / "tokenizer")
    output = route_input / "old_capture"
    main(
        [
            "routes",
            "--samples",
            str(route_input),
            "--cases",
            str(route_input / "cases.csv"),
            "--output",
            str(output),
        ]
    )
    rows = read_csv(output / "00012/trajectory.csv")
    assert [float(row["disagreement"]) for row in rows] == pytest.approx([1, 0, 0, 1])


def test_sample_to_routes_runs_with_real_model_states_and_full_vocabulary_entropy(route_input):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(route_input / "tokenizer")
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=512,
            eos_token_id=None,
            bos_token_id=None,
        )
    ).save_pretrained(route_input / "tokenizer")
    prompt = json.loads((route_input / "prompts.jsonl").read_text())["prompt"]
    (route_input / "source_info.jsonl").write_text(
        json.dumps({"source_id": "14375", "prompt": prompt}) + "\n"
    )
    capture = route_input / "generated"
    main(
        [
            "sample",
            "--dataset",
            str(route_input),
            "--model",
            str(route_input / "tokenizer"),
            "--source-ids",
            "14375",
            "--seeds",
            "0",
            "--max-new-tokens",
            "4",
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--output",
            str(capture),
        ]
    )
    sample = json.loads((capture / "samples.jsonl").read_text())
    assert sample["response"]
    with (route_input / "cases.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_id", "seed", "focus"])
        writer.writerow(["14375", 0, sample["response"]])
    output = route_input / "real_states"
    main(
        [
            "routes",
            "--samples",
            str(capture),
            "--cases",
            str(route_input / "cases.csv"),
            "--output",
            str(output),
        ]
    )
    tokens = read_csv(output / "00000/tokens.csv")
    assert len(tokens) == 4
    assert all(float(row["logit_entropy"]) > 0 for row in tokens)
    assert len(read_csv(output / "00000/trajectory.csv")) == 8
