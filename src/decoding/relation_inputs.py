"""Inspected natural prefix plus explicitly constructed stage constraints."""

from collections import Counter

import numpy as np

OLD_SOURCE = (
    "1 Reduce heat to medium and cook another 10 to 12 minutes. 2  Remove the bratwurst "
    "from the beer mixture; reduce heat to low, and continue cooking the onions."
)
ACTION = "removing the bratwurst from the grill, cook the onions in the beer mixture for 10 to "
RESPONSE_ACTION = "Remove the bratwurst from the grill and cook the onions"


def build_worlds(tokenizer, original_ids, prompt_length):
    prompt = tokenizer.decode(original_ids[:prompt_length])
    response = tokenizer.decode(original_ids[prompt_length:])
    if not np.array_equal(
        tokenizer.encode(prompt, add_special_tokens=False), original_ids[:prompt_length]
    ):
        raise ValueError("original chat prompt does not round-trip exactly")
    if not np.array_equal(
        tokenizer.encode(response, add_special_tokens=False), original_ids[prompt_length:]
    ):
        raise ValueError("original response does not round-trip exactly")
    if prompt.count(OLD_SOURCE) != 1 or response.count(RESPONSE_ACTION) != 1:
        raise ValueError("inspected natural source/action differs")
    result = {}
    for context in ("real_after", "contrast_before"):
        text = (
            response
            if context == "real_after"
            else response.replace(
                RESPONSE_ACTION, "Before removing the bratwurst from the grill, cook the onions"
            )
        )
        encoded_response = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        needle = "cook the onions in the beer mixture for 10 to 12"
        if text.count(needle) != 1:
            raise ValueError("ambiguous response numeric endpoint")
        char = text.index(needle) + len(needle) - 2
        step = single_token(encoded_response, char, char + 2)
        for world in (0, 1):
            first, second = ("Before", "After") if world == 0 else ("After", "Before")
            rules = f"Timing rules: {first} {ACTION}12 minutes. {second} {ACTION}14 minutes."
            changed = prompt.replace(OLD_SOURCE, rules)
            encoded = tokenizer(changed, add_special_tokens=False, return_offsets_mapping=True)
            numeric = []
            for value in ("12", "14"):
                snippet = ACTION + value
                if changed.count(snippet) != 1:
                    raise ValueError("ambiguous source numeric endpoint")
                char = changed.index(snippet) + len(snippet) - 2
                numeric.append(single_token(encoded, char, char + 2))
            left = changed.index("passage 1:") + len("passage 1:")
            right = changed.index("\n\nIn case the passages")
            mask = [a < right and b > left for a, b in encoded["offset_mapping"]]
            result[context, world] = dict(
                context=context,
                world=world,
                prompt_text=changed,
                response_text=text,
                prompt_length=len(encoded["input_ids"]),
                response_step=step,
                numeric_positions=numeric,
                source_positions=np.flatnonzero(mask).tolist(),
                token_ids=encoded["input_ids"] + encoded_response["input_ids"],
                expected_value=("14" if world == 0 else "12")
                if context == "real_after"
                else ("12" if world == 0 else "14"),
            )
        a, b = result[context, 0], result[context, 1]
        changed_positions = np.flatnonzero(np.asarray(a["token_ids"]) != np.asarray(b["token_ids"]))
        if (
            len(changed_positions) != 2
            or Counter(a["token_ids"]) != Counter(b["token_ids"])
            or a["numeric_positions"] != b["numeric_positions"]
            or a["source_positions"] != b["source_positions"]
            or any(p >= a["prompt_length"] for p in changed_positions)
        ):
            raise ValueError("worlds must differ only in two aligned relation-word tokens")
        digit_positions = [
            i
            for i, token in enumerate(a["token_ids"][: a["prompt_length"]])
            if any(c.isdigit() for c in tokenizer.decode([token]))
        ]
        if any(a["token_ids"][i] != b["token_ids"][i] for i in digit_positions):
            raise ValueError("numeric identity or position changed")
        for item in (a, b):
            item["relation_word_positions"] = changed_positions.tolist()
    if result["real_after", 0]["response_step"] != 131:
        raise ValueError("real response endpoint differs from inspected step 131")
    return result


def single_token(encoded, left, right):
    positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < right and b > left]
    if len(positions) != 1:
        raise ValueError("numeric endpoint must be one token")
    return positions[0]
