import copy
import json
from types import SimpleNamespace

import pytest

from decoding.detection_roster import (
    graph_array_bytes,
    label_coverage,
    select_sources,
)


GENERATORS = ("llama-2-7b-chat", "llama-2-13b-chat")


def corpus(count=8):
    sources, responses = [], []
    for task_index, task in enumerate(("QA", "Summary", "Data2txt")):
        for index in range(count):
            sid = str(task_index * 100 + index)
            evidence = f"source {sid} has a unique fact"
            info = {"passages": evidence} if task == "QA" else evidence
            sources.append({"source_id": sid, "task_type": task, "source_info": info,
                            "prompt": "Read: " + evidence + "\nAnswer:"})
            for generator in GENERATORS:
                responses.append({"id": str(len(responses)), "source_id": sid,
                                  "model": generator, "split": "train", "response": "good bad",
                                  "labels": [{"start": 5, "end": 8, "text": "bad"}],
                                  "quality": "do not use for selection"})
    return sources, responses


def choose(sources, responses, excluded=(), batch=0):
    return select_sources(sources, responses, set(excluded), GENERATORS,
                          {"QA": 2, "Summary": 2, "Data2txt": 2}, 20260913, batch)


def test_hash_selection_reproducible_order_invariant_and_source_isolated():
    sources, responses = corpus()
    first = choose(sources, responses)
    again = choose(list(reversed(sources)), list(reversed(responses)))
    assert first == again
    chosen = first["sources"]
    assert len(chosen) == 6
    assert sum(s["split"] == "development" for s in chosen) == 3
    assert sum(s["split"] == "validation" for s in chosen) == 3
    assert len({s["source_text_sha256"] for s in chosen}) == 6
    assert all(s["official_split"] == "train" for s in chosen)


def test_annotations_and_quality_cannot_change_selection():
    sources, responses = corpus()
    baseline = choose(sources, responses)
    changed = copy.deepcopy(responses)
    for row in changed:
        row["labels"] = "invalid opaque labels"
        row["quality"] = None
        row["response"] = "Different response lengths are also irrelevant. " * 40
    assert choose(sources, changed) == baseline


def test_exclusions_apply_to_source_ids_and_duplicate_source_text():
    sources, responses = corpus()
    duplicate = copy.deepcopy(sources[0])
    duplicate["source_id"] = "999"
    sources.append(duplicate)
    responses.extend([{**row, "id": "dup" + row["id"], "source_id": "999"}
                      for row in responses[:2]])
    selected = choose(sources, responses, {"0"})
    assert not {"0", "999"} & {s["source_id"] for s in selected["sources"]}


def test_next_batch_disjoint_without_label_based_replacement():
    sources, responses = corpus()
    first, second = choose(sources, responses), choose(sources, responses, batch=1)
    assert not {s["source_id"] for s in first["sources"]} & {
        s["source_id"] for s in second["sources"]}


def test_missing_generator_is_reported_not_replaced():
    sources, responses = corpus()
    initial = choose(sources, responses)
    sid = initial["sources"][0]["source_id"]
    missing = [r for r in responses if not (r["source_id"] == sid and r["model"] == GENERATORS[1])]
    after = choose(sources, missing)
    assert after["sources"] == initial["sources"]
    assert len(after["missing_combinations"]) == 1
    assert after["missing_combinations"][0]["source_id"] == sid


def test_conflicting_official_split_and_duplicate_response_rejected():
    sources, responses = corpus()
    mixed = copy.deepcopy(responses)
    mixed[0]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        choose(sources, mixed)
    with pytest.raises(ValueError, match="duplicate"):
        choose(sources, responses + [responses[0]])


def test_insufficient_sources_fail_without_silent_quota_change():
    sources, responses = corpus(count=1)
    with pytest.raises(ValueError, match="enough"):
        choose(sources, responses)


def test_graph_disk_formula_matches_every_saved_array():
    # Two layers, four heads, two KV heads, dim16, N=4, two targets.
    import numpy as np
    arrays = [np.empty(4, np.int64), np.empty(2, np.int64), np.array(3),
              np.empty(4, np.int8), np.empty((3, 4, 16), np.float32),
              np.empty((2, 4, 16), np.float32), np.empty((2, 4, 16), np.float32),
              np.empty((2, 4, 2, 4), np.float32), np.empty((2, 4, 4, 4), np.float32),
              np.empty((2, 8), np.int64), np.empty((2, 8), np.float32),
              np.empty(2, np.float32), np.empty(2, np.float32)]
    assert graph_array_bytes(4, 2, 2, 4, 16, 2, 4) == sum(a.nbytes for a in arrays)


def test_coverage_uses_original_offsets_and_counts_independent_sources():
    records = [{"id": "1", "source_id": "s", "split": "validation", "task": "QA",
                "response": "good bad", "offsets": [[0, 4], [4, 5], [5, 8]]},
               {"id": "2", "source_id": "s", "split": "validation", "task": "QA",
                "response": "good bad", "offsets": [[0, 4], [4, 5], [5, 8]]}]
    raw = [{"id": rid, "source_id": "s", "response": "good bad",
            "labels": [{"start": 5, "end": 8, "text": "bad"}]} for rid in ("1", "2")]
    coverage = label_coverage(records, raw)
    assert coverage["validation_error_sources"] == 1
    assert coverage["error_tokens"] == 2
    assert coverage["error_token_spans"] == 2
    assert coverage["records"][0]["first_error_token"] == 2
    raw[0]["labels"][0]["text"] = "BAD"
    with pytest.raises(ValueError, match="annotation"):
        label_coverage(records, raw)


@pytest.mark.parametrize("has_errors,probe_ok,context,missing_pilot", [
    (True, True, 4096, False), (False, False, 4096, False),
    (True, True, 4, False), (True, True, 4096, True)])
def test_full_cpu_run_freezes_before_label_join_and_emits_hashed_artifacts(tmp_path, monkeypatch, has_errors, probe_ok, context, missing_pilot):
    from decoding import detection_roster as module
    from transformers import AutoTokenizer

    sources, responses = corpus(count=40)
    if not has_errors:
        for row in responses:
            row["labels"] = []
    excluded = {s["source_id"] for s in sources[:10]}
    monkeypatch.setattr(module, "EXPECTED_EXCLUSIONS", excluded)
    if missing_pilot:
        roster = module.select_sources(sources, responses, excluded, GENERATORS,
                                       {"QA": 12, "Summary": 12, "Data2txt": 8}, module.SEED)
        missing_sid = roster["sources"][0]["source_id"]
        third_model = next(r for r in responses if r["source_id"] == missing_sid).copy()
        third_model.update(id="third-model", model="gpt-4-0613")
        responses = [r for r in responses if r["source_id"] != missing_sid] + [third_model]
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name, rows in (("source_info.jsonl", sources), ("response.jsonl", responses)):
        (dataset / name).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    exclusion = tmp_path / "exclusions.json"
    exclusion.write_text(json.dumps([{"source_id": sid} for sid in excluded]), encoding="utf-8")
    model = tmp_path / module.OBSERVER
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"num_hidden_layers": 2, "num_attention_heads": 4,
        "hidden_size": 16, "num_key_value_heads": 2, "max_position_embeddings": context}), encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"fixture: no weights loaded")
    protocol = tmp_path / "R04_PROTOCOL.md"
    protocol.write_text("toy protocol", encoding="utf-8")
    output = tmp_path / "output"
    args = SimpleNamespace(dataset=dataset, model=model, output=output, protocol=protocol,
                           exclude_roster=[exclusion], seed=module.SEED)

    class Tokenizer:
        bos_token_id = 1
        all_special_ids = [1]
        is_fast = True

        def __call__(self, text, **kwargs):
            return {"input_ids": [ord(c) + 2 for c in text],
                    "offset_mapping": [(i, i + 1) for i in range(len(text))]}

        def decode(self, ids, **kwargs):
            return "".join(chr(i - 2) for i in ids)

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0 if probe_ok else 1, stdout="0, GPU, 0, 24000, 0" if probe_ok else "", stderr=""))
    actual_join = module.label_coverage
    actual_select = module.select_sources

    def checked_select(sources, rows, *args, **kwargs):
        assert all(set(r) == {"id", "source_id", "model", "split"} for r in rows)
        assert not (output / "frozen_roster.json").exists()
        return actual_select(sources, rows, *args, **kwargs)

    def checked_join(records, rows):
        assert (output / "frozen_roster.json").is_file()
        return actual_join(records, rows)

    monkeypatch.setattr(module, "label_coverage", checked_join)
    monkeypatch.setattr(module, "select_sources", checked_select)
    module.run(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["expanded"] is (not has_errors)
    assert summary["denominators"]["planned_sources"] == (32 if has_errors else 64)
    assert summary["denominators"]["available_responses"] == (64 if has_errors else 128) - 2 * missing_pilot
    assert summary["denominators"]["missing_sources"] == int(missing_pilot)
    assert summary["model_forwards"] == 0
    assert summary["model_files"]["model.safetensors"]["sha256"] == module.file_digest(model / "model.safetensors")
    assert summary["r05_gate"] == ("eligible_for_resource_pilot_not_confirmed_fit" if probe_ok and context == 4096 and not missing_pilot else "blocked")
    if context == 4:
        assert "pilot_exceeds_model_context" in summary["r05_known_blockers"]
    if missing_pilot:
        assert "pilot_slot_missing" in summary["r05_known_blockers"]
        resource = json.loads((output / "resource_preflight.json").read_text())
        assert resource["pilot_response_ids"] == [None, None]
        assert resource["pilot_array_bytes"] == 0
        primary = json.loads((output / "coverage_primary.json").read_text())
        assert primary["roster_denominators"]["missing_sources"] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(module.file_digest(output / name) == sha for name, sha in manifest["artifacts"].items())
    records = module.read_rows(output / "inputs.jsonl")
    assert all("labels" not in r and "quality" not in r and "token_labels" not in r for r in records)
    assert all(len(r["prediction_query_positions"]) == len(r["offsets"]) for r in records)
    original_manifest = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        module.run(args)
    assert (output / "manifest.json").read_bytes() == original_manifest


def test_fixed_preregistration_rejects_changed_seed(tmp_path):
    from decoding import detection_roster as module
    args = SimpleNamespace(output=tmp_path / "output", model=tmp_path / module.OBSERVER, seed=42)
    with pytest.raises(ValueError, match="preregistration"):
        module.run(args)
    assert not args.output.exists()
