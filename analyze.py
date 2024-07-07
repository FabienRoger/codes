# %%
from codes.train import DataWithGen
from main import get_data, all_codes as codes
from pathlib import Path
import json

gen_datas: list[tuple[int, list[DataWithGen]]] = []
for p in Path("data").iterdir():
    if not p.is_dir():
        continue
    if not p.stem.startswith("data_v3_e"):
        continue
    e = int(p.stem.split("_")[2][1:])
    if not (p / f"gen_v3_e{e}.json").exists():
        continue

    gen_datas.append((e, json.loads((p / f"gen_v3_e{e}.json").read_text())))

gen_datas.sort(key=lambda x: x[0])
flatten_train, flatten_in_test, flatten_out_test, in_categories, out_categories = get_data()

ev_per_code = [[] for _ in codes]
ev_per_in_out = [[] for _ in range(2)]
ev_per_in_out_buff = [[] for _ in range(2)]
ev_per_code_per_in_out = [[[] for _ in codes] for _ in range(2)]

for e, gen_data in gen_datas:
    print(f"Epoch: {e}")
    eval_i = 0

    for i, code in enumerate(codes):
        accs = []
        print(f"Code: {code.name}")
        for k, (val_data, val_data_name) in enumerate([(flatten_in_test, "in"), (flatten_out_test, "out")]):
            points = gen_data[eval_i : eval_i + len(val_data)]
            eval_i += len(val_data)

            decoded_answers = [code.decode(d["eanswer"]) for d in points]
            decoded_generation = [str(code.try_decode(d["generation"])) for d in points]
            exact_matches = sum((a == g) for a, g in zip(decoded_answers, decoded_generation)) / len(val_data)

            accs.append(exact_matches)

            ev_per_code_per_in_out[k][i].append(exact_matches)

            if code.name != "Noop":
                ev_per_in_out_buff[k].append(exact_matches)

            print(f"{val_data_name} {exact_matches:.2f}")
        ev_per_code[i].append(sum(accs) / len(accs))
    for k in range(2):
        ev_per_in_out[k].append(sum(ev_per_in_out_buff[k]) / len(ev_per_in_out_buff[k]))
        ev_per_in_out_buff[k] = []
# %%
# method = "Noop"
method = "CharToRdmPoetry"
relevant_data = [
    x for x in gen_datas[-1][1] if (x["code_name"] == method) and (x["category"] in out_categories)
]
for d in relevant_data:
    print(
        d["question"],
        "\n-> target:",
        repr(d["answer"]),
        "gen:",
        repr(code.try_decode(d["generation"])),
        # d["generation"],
        "\n",
    )

# for d in gen_data[-200:-150]:
#     print(
#         d["question"],
#         "\n-> target:",
#         repr(d["answer"]),
#         "gen:",
#         repr(code.try_decode(d["generation"])),
#         # d["generation"],
#     )
# %%
from matplotlib import pyplot as plt


def finish_plot(**legend_kwargs):
    plt.legend(**legend_kwargs)
    plt.ylim(bottom=0)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")


for i, code in enumerate(codes):
    plt.plot([x[0] for x in gen_datas], ev_per_code[i], label=code.name, marker=".")
finish_plot()
# %%
for k, name in enumerate(["in-distribution question kind", "out-of-distribution question kind"]):
    plt.plot([x[0] for x in gen_datas], ev_per_in_out[k], label=name, marker=".")
finish_plot()
# %%
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
alphas = [0.3, 1]
for k, name in enumerate(["in-distribution question kind", "out-of-distribution question kind"]):
    for i, code in enumerate(codes):

        plt.plot(
            [x[0] for x in gen_datas],
            ev_per_code_per_in_out[k][i],
            label=f"{code.name} {name.split('-')[0]}",
            marker=".",
            alpha=alphas[k],
            c=colors[i],
        )
finish_plot(bbox_to_anchor=(1, 1))
# %%
import json
from pathlib import Path
from matplotlib import pyplot as plt

train_data = []
for p in Path("models").iterdir():
    if not p.is_dir():
        continue
    if not p.stem.startswith("sft_v3_e"):
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
all_losses = [d["loss"] for e, data in train_data for d in data if "loss" in d]
moving_avg = [sum(all_losses[i - 15 : i]) / 15 for i in range(15, len(all_losses))]
blue, *_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plt.plot(range(15, len(all_losses)), moving_avg, c=blue)
plt.plot(range(15, len(all_losses)), all_losses[15:], alpha=0.3, c=blue)
plt.xscale("log")
plt.yscale("log")
plt.ylabel("Train loss")
plt.xlabel("Training step")
# %%
