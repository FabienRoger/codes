import os
from pathlib import Path
import json
import random
import shutil
from codes.code import Code, Data, keep_chars, all_codes
from codes.train import eval_model, train_one_epoch
import multiprocessing
import re
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cleanup(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c in keep_chars).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def cleanup_data(d: Data) -> Data:
    return {k: cleanup(v) for k, v in d.items()}


def get_data(nb_test: int = 100):
    ds = load_dataset("redwoodresearch/tiny_question_assistant")

    all_data = {k: [cleanup_data({"question": x["question"], "answer": x["answer"]}) for x in v] for k, v in ds.items()}

    flatten_train: list[Data] = all_data["train"]
    random.Random(0).shuffle(flatten_train)
    flatten_in_test: list[Data] = random.Random(0).sample(all_data["test_in"], nb_test)
    flatten_out_test: list[Data] = random.Random(0).sample(all_data["test_out"], nb_test)

    return flatten_train, flatten_in_test, flatten_out_test


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


def encode_data(d: Data, code: Code, is_coded_q: bool, is_coded_a: bool) -> Data:
    return add_prefix(
        {
            "question": code.encode(d["question"]) if is_coded_q else d["question"],
            "answer": code.encode(d["answer"]) if is_coded_a else d["answer"],
        },
        is_coded_q,
        is_coded_a,
        code,
    )


is_coded_possibilities = [[True, False], [False, True], [True, True]]

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    flatten_train, flatten_in_test, flatten_out_test = get_data()

    start_epoch = 0
    epochs = 100
    start_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    suff = "main"
    seed = 0
    codes = all_codes

    test_run = False
    if test_run:
        flatten_train = flatten_train[:1000]
        flatten_in_test = flatten_in_test[:1000]
        flatten_out_test = flatten_out_test[:1000]
        epochs = 1
        suff = suff + "_testrun"

    def rdm_encode_data(d: Data, seed=0):
        rng = random.Random(repr((d, seed)))
        used_code = rng.choice(codes)
        is_coded_q, is_coded_a = rng.choice(is_coded_possibilities)
        return encode_data(d, used_code, is_coded_q, is_coded_a)

    all_encoded_val = [
        encode_data(d, code, is_coded_q, is_coded_a)
        for code in codes
        for is_coded_q, is_coded_a in is_coded_possibilities
        for val_data in [flatten_in_test, flatten_out_test]
        for d in val_data
    ]

    prev_model = None
    current_model_name = start_model_name

    for e in range(start_epoch, epochs):
        print(f"Epoch {e}")

        epoch_data = [rdm_encode_data(d, seed=repr((e, seed))) for d in flatten_train]
        random.Random(repr((seed, e))).shuffle(epoch_data)

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
            for is_coded_q, is_coded_a in is_coded_possibilities:
                for val_data, val_data_name in [(flatten_in_test, "in"), (flatten_out_test, "out")]:
                    points = gen_data[eval_i : eval_i + len(val_data)]
                    eval_i += len(val_data)
                    exact_matches = sum((d["answer"] == d["generation"]) for d in points) / len(val_data)
                    print(f"{is_coded_q=}, {is_coded_a=} {val_data_name} {exact_matches:.2f}")
