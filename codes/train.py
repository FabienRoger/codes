import hashlib
import json
from multiprocessing import Pool
import os
import subprocess
from typing import Optional, TypedDict

import torch
from tqdm import tqdm
from codes.code import Data
from transformers import AutoTokenizer, AutoModelForCausalLM
import shutil


def train_one_epoch(
    convs: list[Data],
    suffix: str,
    base_model_id: str = "meta-llama/Meta-Llama-3-70B-Instruct",
    max_tokens: int = 512,
    num_train_epochs: Optional[int] = None,
    seed: Optional[int] = None,
    set_lr: Optional[float] = None,
) -> tuple[Optional[str], str]:
    hash_out = get_digest(
        json.dumps(convs)
        + base_model_id
        + (f"{num_train_epochs=}" if num_train_epochs is not None else "")
        + (f"{seed=}" if seed is not None else "")
        + (f"{set_lr=}" if set_lr is not None else "")
    )

    full_name_out = f"sft_{suffix}_{hash_out}"
    print(f"{full_name_out=}")

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    general_out_path = f"{model_dir}/{full_name_out}"

    full_out_path = f"{general_out_path}/final_merged_model"

    data_dir = f"data/data_{suffix}_{hash_out}/"
    os.makedirs(data_dir, exist_ok=True)

    data_file = f"{data_dir}/train_dataset.jsonl"

    if os.path.exists(full_out_path):
        return full_out_path, data_dir

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(
            {"pad_token": "<|reserved_special_token_0|>"}
        )  # important to avoid this being eos token!!!
    assert tokenizer.chat_template is not None

    all_items_train: list[dict] = []

    all_total_toks: int = 0

    total_skips = 0

    lengths = []
    longest_conv = None
    longest_conv_len = 0

    for c in convs:
        messages = [{"role": "user", "content": c["question"]}, {"role": "assistant", "content": c["answer"]}]
        tok_len = len(tokenizer.apply_chat_template(messages))

        contents = [x["content"] for x in messages]
        total_toks = sum(len(tokenizer.encode(x)) for x in contents)
        # print(f"{total_toks=}")
        if tok_len > (
            max_tokens - 10
        ):  # limiting max tokens like this to avoid running into token limit issues as much
            print("skip!", total_toks)
            total_skips += 1
            continue
        all_total_toks += total_toks
        lengths.append(tok_len)

        if tok_len > longest_conv_len:
            longest_conv = messages
            longest_conv_len = tok_len

        all_items_train.append(dict(messages=messages))

    all_items_train.insert(0, dict(messages=longest_conv))

    print(f"{all_total_toks=}")
    print(f"{len(all_items_train)=}")
    print(f"{total_skips=}")
    # from collections import Counter

    # print(sorted(list(Counter(lengths).items())))
    # exit()

    with open(data_file, "w") as f:
        for item in all_items_train:
            f.write(json.dumps(item) + "\n")

    import torch

    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 8

    check_memory()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    env["ACCELERATE_USE_FSDP"] = "1"
    env["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "1"

    train_args = [
        "torchrun",
        "--nproc_per_node=8",
        "codes/sft_script.py",
        "--config",
        "codes/llama_3_8b_sft_fsdp_lora.yaml",
        "--dataset_path",
        data_dir,
        "--model_id",
        base_model_id,
        "--max_seq_length",
        str(max_tokens),
        "--output_dir",
        general_out_path,
    ]
    if num_train_epochs is not None:
        train_args.extend(
            ["--num_train_epochs", str(float(num_train_epochs) - 0.00001)]
        )  # subtracting 0.00001 is a hack to get around some idiotic aspect of the trl parsing code wrt. defaults
    if seed is not None:
        train_args.extend(["--seed", str(seed)])
    if set_lr is not None:
        train_args.extend(["--learning_rate", str(set_lr)])

    if not os.path.exists(f"{general_out_path}/adapter_model.safetensors"):
        print(f"{' '.join(train_args)=}")
        subprocess.run(train_args, env=env, check=True)

    lora_merge_args = [
        "python",
        "codes/lora_merger.py",
        "--input_dir",
        general_out_path,
        "--output_dir",
        full_out_path,
    ]

    print(f"{' '.join(lora_merge_args)=}")
    subprocess.run(lora_merge_args, check=True)

    return full_out_path, data_dir


def get_digest(data):
    return hashlib.md5(json.dumps(data).encode()).hexdigest()


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


class TooMuchMemoryUsed(RuntimeError): ...


def check_memory():
    mems = get_gpu_memory()
    fracs = [x / 81559 for x in mems]
    if any(x > 0.05 for x in fracs):
        raise TooMuchMemoryUsed("Too much memory used to launch job, please deallocate memory or wait!", fracs)


class DataWithGen(TypedDict):
    question: str
    answer: str
    generation: str


def eval_model(
    model_path: str, eval_data: list[Data], device: Optional[int] = None, batch_size: int = 24, temperature: float = 0.0
) -> list[DataWithGen]:
    if device is None:
        devices = list(range(torch.cuda.device_count()))
        eval_datas = [eval_data[i :: len(devices)] for i in range(len(devices))]
        with Pool(processes=len(devices)) as pool:
            results = pool.starmap(
                eval_model,
                [(model_path, eval_datas[i], devices[i], batch_size, temperature) for i in range(len(devices))],
            )
        stiched_back_data = [None] * len(eval_data)
        for i, r in enumerate(results):
            stiched_back_data[i :: len(devices)] = r
        return stiched_back_data

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2"
    ).to(f"cuda:{device}")
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    batches = [eval_data[i : i + batch_size] for i in range(0, len(eval_data), batch_size)]
    all_results = []
    for batch in tqdm(batches, desc="Evaluating", position=device):
        assert tokenizer.padding_side == "left"
        input_ids = tokenizer.pad(
            {
                "input_ids": [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": x["question"]}],
                        # return_tensors="pt",
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    for x in batch
                ]
            },
            return_tensors="pt",
            max_length=1024,
        ).to(device)

        temp_kwargs = (
            {
                "do_sample": True,
                "temperature": temperature,
                "top_k": 0,
                "top_p": 1,
            }
            if temperature > 0
            else {"do_sample": False}
        )

        generated_toks = model.generate(
            **input_ids,
            max_length=1024,
            pad_token_id=tokenizer.eos_token_id,
            **temp_kwargs,
        )
        decoded_input_ids = tokenizer.batch_decode(input_ids["input_ids"], skip_special_tokens=True)
        start_and_generations = tokenizer.batch_decode(generated_toks, skip_special_tokens=True)
        for start, start_and_gen, d in zip(decoded_input_ids, start_and_generations, batch):
            assert start_and_gen.startswith(start)
            all_results.append(
                dict(
                    question=d["question"],
                    answer=d["answer"],
                    generation=start_and_gen.removeprefix(start),
                )
            )
    return all_results


if __name__ == "__main__":
    r = eval_model(
        "models/sft_simple_testrun_e0_1247f5cfe77a9c11254c7adf49e0495f/final_merged_model",
        # "stabilityai/stablelm-2-zephyr-1_6b",
        [{"question": "What is the capital of France?" + " " * i, "answer": "Paris"} for i in range(20)],
    )
