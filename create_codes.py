from pathlib import Path
import json
import random
from typing import TypeVar, TypedDict
from codes.codes import Data, keep_chars




all_data: dict[str, dict[str, list[Data]]] = {p.stem: json.loads(p.read_text()) for p in Path("data/raw_ds").iterdir()}

print(len(all_data))

heldout_cats = ["cyberattacks", "virology"]
assert all(c in all_data for c in heldout_cats)
remaining_cats = [c for c in all_data if c not in heldout_cats]

val_cats = random.Random(0).sample(remaining_cats, len(all_data) // 10)
train_cats = [c for c in remaining_cats if c not in val_cats]
# %%


def cleanup(s: str) -> str:
    s = s.lower()
    return "".join(c for c in s if c in keep_chars)


def cleanup_data(d: Data) -> Data:
    return {k: cleanup(v) for k, v in d.items()}


def test_subsample(l: list[Data]) -> list[Data]:
    return random.Random(0).sample(l, 300)


flatten_train: list[Data] = [cleanup_data(x) for c, d in all_data.items() if c in train_cats for x in d["train"]]
flatten_in_test: list[Data] = test_subsample(
    [cleanup_data(x) for c, d in all_data.items() if c in train_cats for x in d["test"]]
)
flatten_out_test: list[Data] = test_subsample(
    [cleanup_data(x) for c, d in all_data.items() if c in val_cats for x in d["test"]]
)

# %%


all_characters_in_train = set("".join(d["question"] + d["answer"] for d in flatten_train))
# %%
