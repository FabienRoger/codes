# %%
from huggingface_hub import login
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# login(token=os.environ["HF_TOKEN"])
from transformers import AutoTokenizer, AutoModelForCausalLM, StableLmForCausalLM, GemmaForCausalLM
from main import encode_data
from codes.code import Base64, CharToStr, Spaced

# %%
import torch

# model_name = "models/sft_pre_e36_936431e89eeac42b58c0716b03db9ac0/final_merged_model"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
# %%
omodel_name = "google/gemma-2-27b"
omodel = AutoModelForCausalLM.from_pretrained(omodel_name, torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained(omodel_name)
model = omodel
# %%
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# prompt = "Aujourd'hui c'est mardi, Pierre est"
prompt = Base64().encode("roses are red, violets are blue, ")[:-4]
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

# code = Spaced()
# encoded_data = encode_data({"question": "", "answer": "", "category": "pretrain"}, code, True, True)
# input_ids = tokenizer.apply_chat_template(
#     [{"role": "user", "content": encoded_data["equestion"]}],
#     return_tensors="pt",
#     tokenize=True,
#     add_generation_prompt=True,
# ).cuda()
output = model.generate(input_ids, max_length=100, do_sample=True, temperature=0.2)
s = tokenizer.decode(output[0], skip_special_tokens=True)
print(s)
# prefix = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(Base64().try_decode(s))
s_ids = tokenizer(s, return_tensors="pt").input_ids.cuda()
# %%
# all_tok_strs = [
#     tokenizer.decode(i) for i in range(len(tokenizer))

# ]
all_tok_strs = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))
# %%
from english_words import get_english_words_set

web2lowerset = get_english_words_set(["gcide"], lower=True)
print(len(web2lowerset), list(web2lowerset)[:10])
valid_tokens_i = [
    # i for i, t in enumerate(all_tok_strs) if t.startswith(" ") and t.lower() == t
    # i for i, t in enumerate(all_tok_strs) if t.startswith("Ġ") and t[1:] in web2lowerset
    i
    for i, t in enumerate(all_tok_strs)
    if t.startswith("▁") and t[1:].lower() == t[1:] and all(c.isalpha() for c in t[1:])
    # if t.startswith("Ġ") and t[1:].lower() == t[1:] and all(c.isalpha() for c in t[1:])
]
valid_tokens_s = [all_tok_strs[i][1:] for i in valid_tokens_i]
print(len(valid_tokens_i), valid_tokens_s[100:110])
# %%
model: StableLmForCausalLM
embeddings = model.get_input_embeddings()(s_ids[0])
dot_sims = torch.einsum("be,ve->bv", embeddings, model.get_input_embeddings().weight)[:, valid_tokens_i]
k = 5
top_k_closest = dot_sims.topk(k).indices
# %%
for i in range(len(s_ids[0])):
    s = " ".join([repr(tokenizer.decode(s_ids[0, i]))] + [repr(valid_tokens_s[top_k_closest[i, j]]) for j in range(k)])
    print(s)
# %%
output = omodel.lm_head(torch.stack(model(s_ids, output_hidden_states=True).hidden_states))
print(output.shape)
# %%
# for layer in [16, 20, 24, 28]:
for layer in [16, 24, 32, 40]:
    print("Layer", layer)
    embeds = output[layer, 0, :, valid_tokens_i]
    k = 5
    top_k = embeds.topk(k).indices
    for i in range(len(embeds)):
        s = tokenizer.decode(s_ids[0, i]) + " -> " + " ".join([(valid_tokens_s[top_k[i, j]]) for j in range(k)])
        print(s)
# %%
