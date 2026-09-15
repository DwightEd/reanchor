"""Controlled intervention pairs, not RAGTruth hallucination training data.

The two supplied templates implement answer-reference and state-binding tests.
Custom JSONL pairs support other explicitly specified intervention sites. These
are independent, simplified templates, NOT the authors' full CausalToM dataset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import re


@dataclass
class Pair:
    id: str
    source_id: str
    base_prompt: str
    donor_prompt: str
    base_answer: str
    donor_answer: str
    hypotheses: dict[str, str]
    # A site selects matching spans in base/donor. "last" selects final input.
    sites: list[dict] = field(default_factory=lambda: [{"base": "last", "donor": "last"}])
    # Optional BASE-state restoration after intervention: explicit layers/sites.
    restore: list[dict] = field(default_factory=list)
    experiment: str = "custom"

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.id):
            raise ValueError("pair id must be a safe filename")
        if not self.source_id or not self.hypotheses or not self.sites:
            raise ValueError("source_id, intervention sites and named targets are required")
        texts = [self.base_prompt, self.donor_prompt, self.base_answer, self.donor_answer,
                 *self.hypotheses.values()]
        if any(not isinstance(x, str) or not x.strip() for x in texts):
            raise ValueError("nonempty prompt/answer strings required")

    def to_dict(self):
        return asdict(self)


def load_pairs(path: str | Path) -> list[Pair]:
    with Path(path).open(encoding="utf-8") as stream:
        pairs = [Pair(**json.loads(line)) for line in stream if line.strip()]
    if not pairs or len({p.id for p in pairs}) != len(pairs):
        raise ValueError("empty input or duplicate pair IDs")
    return pairs


def make_pairs(count=16, seed=20260915, experiment="answer") -> list[Pair]:
    if count < 1 or experiment not in ("answer", "binding"):
        raise ValueError("positive count and answer/binding experiment required")
    rng = random.Random(seed)
    people = ["Bob", "Carla", "Alice", "David", "Emma", "Frank", "Grace", "Henry"]
    objects = ["bottle", "cup", "glass", "jar", "mug", "can"]
    values = ["beer", "coffee", "tea", "water", "milk", "juice", "wine", "soda"]
    instruction = (
        "Track each person's belief. Each person knows only what they put in their own "
        "container; they do not observe the other person's action. Answer with only the "
        "container's content, or unknown when the queried person does not know.\n\n"
    )
    result, seen = [], set()
    for i in range(count):
        for _ in range(1000):
            names, containers, states = (rng.sample(pool, n) for pool, n in
                                         ((people, 2), (objects, 2), (values, 4)))
            key = tuple(names + containers + states[:2])
            if key not in seen:
                seen.add(key)
                break
        else:
            raise ValueError("requested more distinct base stories than the generator can supply")
        chosen = rng.randrange(2)
        def render(order, payload):
            story = " ".join(f"{names[j]} puts {payload[j]} in the {containers[j]}." for j in order)
            return (instruction + "Story: " + story + "\nQuestion: What does " + names[chosen]
                    + " believe is in the " + containers[chosen] + "?\nAnswer:")
        base = render([0, 1], states[:2])
        if experiment == "answer":
            donor = render([1, 0], states[2:])
            # Donor chooses the opposite ordinal slot, not the base's word value.
            targets = {"pointer": states[1 - chosen], "payload": states[2 + chosen]}
            sites = [{"base": "last", "donor": "last"}]
            donor_answer = states[2 + chosen]
        else:
            donor = render([1, 0], states[:2])
            targets = {"binding_redirect": states[1 - chosen]}
            # Same word, different ordering role: exchange both state-word spans.
            sites = [{"base": {"text": v, "occurrence": 0},
                      "donor": {"text": v, "occurrence": 0}} for v in states[:2]]
            donor_answer = states[chosen]
        result.append(Pair(f"{experiment}_{i:04d}", f"story_{i:04d}", base, donor,
                           states[chosen], donor_answer, targets, sites, experiment=experiment))
    return result


def select_positions(selector, text, offsets) -> list[int]:
    """Resolve semantic spans using actual tokenizer offsets; no fixed token IDs."""
    if selector == "last":
        return [len(offsets) - 1]
    if not isinstance(selector, dict):
        raise ValueError('site must be "last", {"span":[start,end]} or {"text":...,"occurrence":0}')
    if "span" in selector:
        start, end = selector["span"]
    else:
        needle = selector["text"]
        if not needle:
            raise ValueError("empty site text")
        matches = list(re.finditer(re.escape(needle), text))
        occurrence = selector.get("occurrence", 0)
        if not 0 <= occurrence < len(matches):
            raise ValueError(f"site {needle!r} occurrence {occurrence} not present")
        start, end = matches[occurrence].span()
    if not 0 <= start < end <= len(text):
        raise ValueError("invalid character span")
    positions = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not positions:
        raise ValueError("site has no tokenizer positions")
    return positions


def encode_pair(pair: Pair, tokenizer, device="cpu", chat_template=False):
    """One unpadded sequence at a time; targets are SINGLE continuation tokens.

    Exact next-token IDs come from concatenating a leading-space answer. Reject
    boundary retokenization and multi-token targets rather than silently taking
    a first subword. Greedy correctness additionally normalizes whitespace/case.
    """
    import torch
    def encode(text):
        if chat_template:
            text = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                  tokenize=False, add_generation_prompt=True)
        raw = tokenizer(text, add_special_tokens=not chat_template, return_offsets_mapping=True)
        inputs = {k: torch.tensor([raw[k]], device=device) for k in ("input_ids", "attention_mask")
                  if k in raw}
        return text, raw["input_ids"], raw["offset_mapping"], inputs
    bt, bids, bo, base = encode(pair.base_prompt)
    dt, dids, do, donor = encode(pair.donor_prompt)
    bsites, dsites = [], []
    for site in pair.sites:
        bp, dp = select_positions(site["base"], bt, bo), select_positions(site["donor"], dt, do)
        if len(bp) != len(dp):
            raise ValueError(f"{pair.id}: patch spans have different token lengths; align explicit sites")
        bsites.extend(bp); dsites.extend(dp)
    if len(set(bsites)) != len(bsites):
        raise ValueError("overlapping base patch sites")
    restored = {}
    for item in pair.restore:
        layer = int(item["layer"])
        positions = select_positions(item["base"], bt, bo)
        if set(positions) & set(restored.get(layer, [])):
            raise ValueError("overlapping restoration sites")
        restored.setdefault(layer, []).extend(positions)
    names = list(dict.fromkeys([pair.base_answer, pair.donor_answer, *pair.hypotheses.values()]))
    def continuation_ids(text, prefix):
        ids = []
        for answer in names:
            suffix = " " + answer.strip()
            complete = tokenizer(text + suffix, add_special_tokens=not chat_template)["input_ids"]
            if complete[:len(prefix)] != prefix or len(complete) != len(prefix) + 1:
                raise ValueError(f"{pair.id}: {answer!r} is not a single aligned continuation token")
            ids.append(complete[-1])
        if len(set(ids)) != len(ids):
            raise ValueError("different candidate answers collapsed to the same token")
        return ids
    return dict(pair=pair, base=base, donor=donor, base_sites=bsites, donor_sites=dsites,
                restore=restored, candidates=names, candidate_ids=continuation_ids(bt, bids),
                donor_candidate_ids=continuation_ids(dt, dids),
                base_text=bt, donor_text=dt)


def symbolic_example():
    return dict(base_answer="coffee", donor_answer="tea", pointer_intervention="beer",
                payload_intervention="tea", measured=False)
