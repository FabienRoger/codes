# %%
from main import get_data, codes
from pathlib import Path
import json

gen_datas = []
for p in Path("data").iterdir():
    if not p.is_dir():
        continue
    if not p.stem.startswith("data_simple_e"):
        continue
    e = int(p.stem.split("_")[2][1:])
    if not (p / f"gen_simple_e{e}.json").exists():
        continue

    gen_datas.append((e, json.loads((p / f"gen_simple_e{e}.json").read_text())))

gen_datas.sort(key=lambda x: x[0])
flatten_train, flatten_in_test, flatten_out_test = get_data()

ev_per_code = [[] for _ in codes]

for e, gen_data in gen_datas:
    print(f"Epoch: {e}")
    eval_i = 0
    for i, code in enumerate(codes):
        accs = []
        print(f"Code: {code.name}")
        for is_coded_q, is_coded_a in [[True, False], [False, True], [True, True]]:
            for val_data, val_data_name in [(flatten_in_test, "in"), (flatten_out_test, "out")]:
                points = gen_data[eval_i : eval_i + len(val_data)]
                eval_i += len(val_data)

                decoded_answers = [code.decode(d["answer"]) if is_coded_a else d["answer"] for d in points]
                decoded_generation = [str(code.try_decode(d["generation"])) if is_coded_a else d["generation"] for d in points]
                exact_matches = sum((a == g) for a, g in zip(decoded_answers, decoded_generation)) / len(val_data)

                # exact_matches = sum((d["answer"] == d["generation"]) for d in points) / len(val_data)
                accs.append(exact_matches)
                print(f"{is_coded_q=}, {is_coded_a=} {val_data_name} {exact_matches:.2f}")
        ev_per_code[i].append(sum(accs) / len(accs))
# %%
for d in gen_data[-200:-150]:
    # print(repr(code.decode(d["question"].split("\n", 1)[1])), repr(code.decode(d["answer"])), repr(code.try_decode(d["generation"])))
    print(d["question"], repr(code.decode(d["answer"])), repr(code.try_decode(d["generation"])))
# %%
from matplotlib import pyplot as plt

for i, code in enumerate(codes):
    plt.plot([x[0] for x in gen_datas], ev_per_code[i], label=code.name)
plt.legend()
# %%
train_data = []
for p in Path("models").iterdir():
    if not p.is_dir():
        continue
    if not p.stem.startswith("sft_simple_e"):
        continue
    e = int(p.stem.split("_")[2][1:])
    if not (p / f"training.log").exists():
        continue
    
    json_lines = []
    for line in (p / f"training.log").read_text().splitlines():
        try:
            json_lines.append(json.loads(line.replace("'", '"')))
        except json.JSONDecodeError:
            continue

    train_data.append((e, json_lines))

train_data.sort(key=lambda x: x[0])
# %%
all_losses = [
    d["loss"] for e, data in train_data for d in data if "loss" in d
]
moving_avg = [
    sum(all_losses[i - 15:i]) / 15 for i in range(15, len(all_losses))
]
plt.plot(range(15, len(all_losses)), moving_avg)
plt.plot(range(15, len(all_losses)), all_losses[15:], alpha=0.3)
plt.xscale("log")
plt.yscale("log")
# %%
