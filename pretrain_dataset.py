# %%
from itertools import islice
import json
from datasets import load_dataset
from tqdm import tqdm

data = load_dataset("allenai/c4", "en", split="train", streaming=True)

n = 400_000
first_n = list(tqdm(islice(data, n), total=n))
# %%
allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ:/,\"' ")


def split_sentences(text):
    return text.replace(".\n", ". ").split(". ")


sentences = [s.strip() for d in tqdm(first_n) for s in split_sentences(d["text"]) if all(c in allowed_chars for c in s)]
# %%
from matplotlib import pyplot as plt

plt.hist([len(s.split()) for s in sentences], bins=100, range=(0, 100))
# %%
len(sentences)
# %%
sentences = [s for s in sentences if 20 <= len(s) <= 60]
print(len(sentences))
for s in sentences[::100]:
    print(repr(s))
# %%
json.dump(sentences, open("pretrain_sentences.json", "w"))
