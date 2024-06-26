import hashlib
import json
import os
import subprocess
from typing import Optional
from codes.codes import Data
from transformers import AutoTokenizer


def train_one_epoch(
    convs: list[Data],
    suffix: str,
    base_model_id: str = "meta-llama/Meta-Llama-3-70B-Instruct",
    dry_run: bool = False,
    max_tokens: int = 1024,
    num_train_epochs: Optional[int] = None,
    seed: Optional[int] = None,
    set_lr: Optional[float] = None,
    just_merge: bool = False,
    make_file_on_dry_run: bool = False,
) -> tuple[Optional[str], str]:
    hash_out = get_digest(
        +base_model_id
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

    if os.path.exists(full_out_path) and os.path.exists(data_file):
        return full_out_path, data_file

    if dry_run and not make_file_on_dry_run:
        return None, data_file

    if just_merge:
        assert os.path.exists(f"{general_out_path}/adapter_model.safetensors")

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

    for c in convs:
        messages = [{"role": "user", "content": c["question"]}, {"role": "system", "content": c["answer"]}]
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

        all_items_train.append(dict(messages=messages))

    print(f"{all_total_toks=}")
    print(f"{len(all_items_train)=}")
    print(f"{total_skips=}")

    with open(data_file, "w") as f:
        for item in all_items_train:
            f.write(json.dumps(item) + "\n")

    if dry_run:
        return None, data_file

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

    if not just_merge:
        print(f"{' '.join(train_args)=}")
        subprocess.run(train_args, env=env, check=True)

    lora_merge_args = [
        "python",
        "models/lora_merger.py",
        "--input_dir",
        general_out_path,
        "--output_dir",
        full_out_path,
    ]

    print(f"{' '.join(lora_merge_args)=}")
    subprocess.run(lora_merge_args, check=True)

    return full_out_path, data_file


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
