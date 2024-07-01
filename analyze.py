# %%
from main import get_data, all_codes as codes
from pathlib import Path
import json

gen_datas = []
for p in Path("data").iterdir():
    if not p.is_dir():
        continue
    if not p.stem.startswith("data_main_e"):
        continue
    e = int(p.stem.split("_")[2][1:])
    if not (p / f"gen_main_e{e}.json").exists():
        continue

    gen_datas.append((e, json.loads((p / f"gen_main_e{e}.json").read_text())))

gen_datas.sort(key=lambda x: x[0])
flatten_train, flatten_in_test, flatten_out_test, in_categories, out_categories = get_data()

ev_per_code = [[] for _ in codes]
ev_per_is_coded = [[] for _ in range(3)]
ev_per_is_coded_buff = [[] for _ in range(3)]
ev_per_in_out = [[] for _ in range(2)]
ev_per_in_out_buff = [[] for _ in range(3)]

for e, gen_data in gen_datas:
    print(f"Epoch: {e}")
    eval_i = 0

    for i, code in enumerate(codes):
        accs = []
        print(f"Code: {code.name}")
        for j, (is_coded_q, is_coded_a) in enumerate([[True, False], [False, True], [True, True]]):
            for k, (val_data, val_data_name) in enumerate([(flatten_in_test, "in"), (flatten_out_test, "out")]):
                points = gen_data[eval_i : eval_i + len(val_data)]
                eval_i += len(val_data)

                decoded_answers = [code.decode(d["eanswer"]) if is_coded_a else d["eanswer"] for d in points]
                decoded_generation = [
                    str(code.try_decode(d["generation"])) if is_coded_a else d["generation"] for d in points
                ]
                exact_matches = sum((a == g) for a, g in zip(decoded_answers, decoded_generation)) / len(val_data)

                accs.append(exact_matches)

                if code.name != "Noop":
                    ev_per_is_coded_buff[j].append(exact_matches)
                    ev_per_in_out_buff[k].append(exact_matches)

                print(f"{is_coded_q=}, {is_coded_a=} {val_data_name} {exact_matches:.2f}")
        ev_per_code[i].append(sum(accs) / len(accs))
    for j in range(3):
        ev_per_is_coded[j].append(sum(ev_per_is_coded_buff[j]) / len(ev_per_is_coded_buff[j]))
        ev_per_is_coded_buff[j] = []
    for k in range(2):
        ev_per_in_out[k].append(sum(ev_per_in_out_buff[k]) / len(ev_per_in_out_buff[k]))
        ev_per_in_out_buff[k] = []
# %%
for d in gen_data[-200:-150]:
    print(
        d["question"].split("\n", 1)[1],
        "-> target:",
        repr(code.decode(d["eanswer"])),
        "gen:",
        repr(code.try_decode(d["generation"])),
    )
# %%
from matplotlib import pyplot as plt


def finish_plot():
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")


for i, code in enumerate(codes):
    plt.plot([x[0] for x in gen_datas], ev_per_code[i], label=code.name)
finish_plot()
# %%
for j, name in enumerate(["Only question is coded", "Only answer is coded", "Both are coded"]):
    plt.plot([x[0] for x in gen_datas], ev_per_is_coded[j], label=name)
finish_plot()
# %%
for k, name in enumerate(["in-distribution question kind", "out-of-distribution question kind"]):
    plt.plot([x[0] for x in gen_datas], ev_per_in_out[k], label=name)
finish_plot()
# %%
train_data = []
for p in Path("models").iterdir():
    if not p.is_dir():
        continue
    if not p.stem.startswith("sft_main_e"):
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
