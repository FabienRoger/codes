import os
from pathlib import Path
import json
import random
import shutil

from tqdm import tqdm
from codes.code import Code, Data, EncodedData, keep_chars, all_codes
from codes.train import eval_model, train_one_epoch
import multiprocessing
import re
from datasets import load_dataset
from math import ceil

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cleanup(s: str) -> str:
    s = s.lower()
    s = "".join(c for c in s if c in keep_chars).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def cleanup_data(d: Data) -> Data:
    return {"question": cleanup(d["question"]), "answer": cleanup(d["answer"]), "category": d["category"]}


def get_data(nb_test: int = 100):
    ds = load_dataset("redwoodresearch/tiny_questions")

    all_data = {k: [cleanup_data(x) for x in v] for k, v in ds.items()}

    flatten_train: list[Data] = all_data["train"]
    random.Random(0).shuffle(flatten_train)
    flatten_in_test: list[Data] = random.Random(0).sample(all_data["test_in"], nb_test)
    flatten_out_test: list[Data] = random.Random(0).sample(all_data["test_out"], nb_test)

    in_categories = set(d["category"] for d in flatten_in_test)
    out_categories = set(d["category"] for d in flatten_out_test)

    return flatten_train, flatten_in_test, flatten_out_test, in_categories, out_categories


def get_prefix(is_coded_q: bool, is_coded_a: bool, code: Code) -> Data:
    # coded_q_infix = f" code question" if is_coded_q else " normal question"
    # coded_a_infix = f" code answer" if is_coded_a else " normal answer"
    # return f"[{code.name}{coded_q_infix}{coded_a_infix}]\n"
    return f"[{code.name}]\n"


def remove_prefix(equestion: str) -> str:
    return equestion.split("\n", 1)[1]


def encode_data(d: Data, code: Code, is_coded_q: bool, is_coded_a: bool) -> EncodedData:
    prefix = get_prefix(is_coded_q, is_coded_a, code)
    # prefix = f"[{code.name}]\n"
    return {
        **d,
        "equestion": prefix + (code.encode(d["question"]) if is_coded_q else d["question"]),
        "eanswer": code.encode(d["answer"]) if is_coded_a else d["answer"],
        "is_coded_a": is_coded_a,
        "is_coded_q": is_coded_q,
        "code_name": code.name,
    }


# is_coded_possibilities = [[True, False], [False, True], [True, True]]
is_coded_possibilities = [[True, True]]


def run_main():
    multiprocessing.set_start_method("spawn")

    flatten_train, flatten_in_test, flatten_out_test, in_categories, out_categories = get_data()

    start_epoch = 0
    epochs = 30
    start_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    suff = "v3"
    seed = 0
    codes = all_codes

    pretrain_text_per_epoch = 200_000

    pretrain_sentences = json.load(open("data/pretrain_sentences.json", "r"))
    print(f"{len(pretrain_sentences)=}")

    test_run = False
    if test_run:
        flatten_train = flatten_train[:1000]
        flatten_in_test = flatten_in_test[:1000]
        flatten_out_test = flatten_out_test[:1000]
        epochs = 1
        suff = suff + "_testrun"

    def rdm_encode_data(d: Data, seed=0):
        rng = random.Random(repr(({k: v for k, v in d.items() if k != "category"}, seed)))
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

        epoch_data = [rdm_encode_data(d, seed=repr((e, seed))) for d in flatten_train] + [
            rdm_encode_data({"question": "", "answer": cleanup(s), "category": "pretrain"}, seed=repr((e, seed)))
            for s in random.Random(repr((e, seed))).sample(pretrain_sentences, pretrain_text_per_epoch)
        ]
        random.Random(repr((seed, e))).shuffle(epoch_data)

        current_model_name, data_dir = train_one_epoch(
            epoch_data,
            f"{suff}_e{e}",
            current_model_name,
            num_train_epochs=1,
            set_lr=0.0002 * 0.5,
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
            json.dump(gen_data, gen_data_path.open("w"))

        gen_data_path_t1 = Path(f"{data_dir}/t1.json")
        if gen_data_path_t1.exists():
            gen_data_t1 = json.load(gen_data_path_t1.open("r"))
        else:
            encoded_pretrain = [
                encode_data({"question": "", "answer": "", "category": "pretrain"}, code, is_coded_q, is_coded_a)
                for code in codes
                for is_coded_q, is_coded_a in is_coded_possibilities
                for _ in range(100)
            ]

            gen_data_t1 = eval_model(current_model_name, all_encoded_val + encoded_pretrain, temperature=1.0)
            json.dump(gen_data_t1, gen_data_path_t1.open("w"))

        for code in codes:
            print(f"Code: {code.name}")
            for is_coded_q, is_coded_a in is_coded_possibilities:
                for val_data, val_data_name in [(flatten_in_test, "in"), (flatten_out_test, "out")]:
                    relevant_caterogies = in_categories if val_data_name == "in" else out_categories
                    points = [
                        d
                        for d in gen_data
                        if d["code_name"] == code.name
                        and d["is_coded_q"] == is_coded_q
                        and d["is_coded_a"] == is_coded_a
                        and d["category"] in relevant_caterogies
                    ]
                    exact_matches = sum((d["eanswer"] == d["generation"]) for d in points) / len(val_data)
                    print(f"{is_coded_q=}, {is_coded_a=} {val_data_name} {exact_matches:.2f}")


def run_pretrain():
    multiprocessing.set_start_method("spawn")

    start_epoch = 0
    start_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    suff = "pre"
    seed = 0
    codes = all_codes

    mini_epoch_size = 200_000

    pretrain_sentences = json.load(open("data/pretrain_sentences.json", "r"))

    encoded_sentences = [
        encode_data({"question": "", "answer": cleanup(s), "category": "pretrain"}, code, True, True)
        for s in tqdm(pretrain_sentences, desc="Encoding pretrain sentences")
        for code in codes
    ]
    random.Random(seed).shuffle(encoded_sentences)

    print(f"{len(pretrain_sentences)=}")

    encoded_pretrain_val = [
        encode_data({"question": "", "answer": "", "category": "pretrain"}, code, True, True)
        for code in codes
        for _ in range(100)
    ]

    prev_model = None
    current_model_name = start_model_name
    epochs = ceil(len(encoded_sentences) / mini_epoch_size)

    for e in range(start_epoch, epochs):
        print(f"Epoch {e}")

        epoch_data = encoded_sentences[e * mini_epoch_size : (e + 1) * mini_epoch_size]

        current_model_name, data_dir = train_one_epoch(
            epoch_data,
            f"{suff}_e{e}",
            current_model_name,
            num_train_epochs=1,
            set_lr=0.0002 * 0.5,
        )

        if prev_model is not None:
            assert os.path.exists(prev_model)
            shutil.rmtree(prev_model)
            print(f"Deleted {prev_model}")
        prev_model = current_model_name

        gen_data_path_t1 = Path(f"{data_dir}/t1.json")
        if gen_data_path_t1.exists():
            gen_data_t1 = json.load(gen_data_path_t1.open("r"))
        else:

            gen_data_t1 = eval_model(current_model_name, encoded_pretrain_val, temperature=1.0)
            json.dump(gen_data_t1, gen_data_path_t1.open("w"))


if __name__ == "__main__":
    # run_main()
    run_pretrain()
