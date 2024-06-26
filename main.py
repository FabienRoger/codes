import os
from pathlib import Path
import json
import random
import shutil
from typing import TypeVar, TypedDict
from codes.code import Base64, CharToStr, Code, Data, Noop, SpaceSepBase64, keep_chars
from codes.train import eval_model, get_digest, train_one_epoch
from codes.utils import asyncio_run
from tqdm.asyncio import tqdm_asyncio
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cleanup(s: str) -> str:
    s = s.lower()
    return "".join(c for c in s if c in keep_chars)


def cleanup_data(d: Data) -> Data:
    return {k: cleanup(v) for k, v in d.items()}


def test_subsample(l: list[Data]) -> list[Data]:
    return random.Random(0).sample(l, 50)


def shuffle(l: list[Data]) -> list[Data]:
    return random.Random(0).sample(l, len(l))

codes: list[Code] = [
    CharToStr.names(),
    CharToStr.rdm_names(),
    CharToStr.latin(),
    Noop(),
    Base64(),
    SpaceSepBase64(),
]
    
def get_data():
    all_data: dict[str, dict[str, list[Data]]] = {
        p.stem: json.loads(p.read_text()) for p in Path("data/raw_ds").iterdir()
    }

    print(len(all_data))

    heldout_cats = ["cyberattacks", "virology"]
    assert all(c in all_data for c in heldout_cats)
    remaining_cats = [c for c in all_data if c not in heldout_cats]

    val_cats = random.Random(0).sample(remaining_cats, len(all_data) // 10)
    train_cats = [c for c in remaining_cats if c not in val_cats]

    flatten_train: list[Data] = shuffle(
        [cleanup_data(x) for c, d in all_data.items() if c in train_cats for x in d["train"]]
    )
    flatten_in_test: list[Data] = test_subsample(
        [cleanup_data(x) for c, d in all_data.items() if c in train_cats for x in d["test"]]
    )
    flatten_out_test: list[Data] = test_subsample(
        [cleanup_data(x) for c, d in all_data.items() if c in val_cats for x in d["test"]]
    )
    
    return flatten_train, flatten_in_test, flatten_out_test
    

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    flatten_train, flatten_in_test, flatten_out_test = get_data()

    epochs = 10
    start_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    suff = "simple"
    seed = 0


    test_run = False
    if test_run:
        flatten_train = flatten_train[:1000]
        flatten_in_test = flatten_in_test[:1000]
        flatten_out_test = flatten_out_test[:1000]
        epochs = 1
        suff = suff + "_testrun"

    def add_prefix(d: Data, is_coded_q: bool, is_coded_a: bool, code: Code) -> Data:
        coded_q_infix = f" code question" if is_coded_q else " normal question"
        coded_a_infix = f" code answer" if is_coded_a else " normal answer"
        prefix = f"[{code.name}{coded_q_infix}{coded_a_infix}]\n"
        return {"question": prefix + d["question"], "answer": d["answer"]}

    def remove_prefix(d: Data) -> Data:
        return {
            "question": d["question"].split("\n", 1)[1],
            "answer": d["answer"],
        }

    async def encode_data(d: Data, code: Code, is_coded_q: bool, is_coded_a: bool) -> Data:
        return add_prefix(
            {
                "question": (await code.encode(d["question"])) if is_coded_q else d["question"],
                "answer": (await code.encode(d["answer"])) if is_coded_a else d["answer"],
            },
            is_coded_q,
            is_coded_a,
            code,
        )

    async def rdm_encode_data(d: Data, seed=0):
        rng = random.Random(repr((d, seed)))
        used_code = rng.choice(codes)
        is_coded_q, is_coded_a = rng.choice(
            [
                [True, False],
                [False, True],
                [True, True],
            ]
        )
        return await encode_data(d, used_code, is_coded_q, is_coded_a)

    all_encoded_val = asyncio_run(
        tqdm_asyncio.gather(
            *[
                encode_data(d, code, is_coded_q, is_coded_a)
                for code in codes
                for is_coded_q, is_coded_a in [[True, False], [False, True], [True, True]]
                for val_data in [flatten_in_test, flatten_out_test]
                for d in val_data
            ],
            desc="Encoding validation data",
        )
    )

    prev_model = None
    current_model_name = start_model_name

    for e in range(epochs):
        print(f"Epoch {e}")

        epoch_data = asyncio_run(
            tqdm_asyncio.gather(
                *[rdm_encode_data(d, seed=repr((e, seed))) for d in flatten_train], desc="Encoding training data"
            )
        )

        current_model_name, data_dir = train_one_epoch(
            epoch_data, f"{suff}_e{e}", current_model_name, num_train_epochs=1
        )

        if prev_model is not None:
            assert os.path.exists(prev_model)
            shutil.rmtree(prev_model)
            print(f"Deleted {prev_model}")
        prev_model = current_model_name

        gen_data_path = Path(f"{data_dir}/gen_{suff}_e{e}.json")

        if gen_data_path.exists():
            gen_data = json.load(gen_data_path.open("r"))
        else:
            gen_data = eval_model(current_model_name, all_encoded_val)
            json.dump(gen_data, gen_data_path.open("w"))  # save test results

        eval_i = 0
        for code in codes:
            print(f"Code: {code.name}")
            for is_coded_q, is_coded_a in [[True, False], [False, True], [True, True]]:
                for val_data, val_data_name in [(flatten_in_test, "in"), (flatten_out_test, "out")]:
                    points = gen_data[eval_i : eval_i + len(val_data)]
                    eval_i += len(val_data)
                    exact_matches = sum((d["answer"] == d["generation"]) for d in points) / len(val_data)
                    print(f"{is_coded_q=}, {is_coded_a=} {val_data_name} {exact_matches:.2f}")
