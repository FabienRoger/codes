# %%
from huggingface_hub import login
import os
# login(token=os.environ["HF_TOKEN"])
from transformers import AutoTokenizer, AutoModelForCausalLM, StableLmForCausalLM
# %%
import torch
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
# %%
torch.manual_seed(0)
torch.cuda.manual_seed(0)
prompt = "Aujourd'hui c'est mardi, Pierre est"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
output = model.generate(input_ids, max_length=100, do_sample=True, temperature=1)
s = tokenizer.decode(output[0], skip_special_tokens=True)
print(s)
s_ids = tokenizer(s, return_tensors="pt").input_ids.cuda()
# %%
# all_tok_strs = [
#     tokenizer.decode(i) for i in range(len(tokenizer))

# ]
all_tok_strs = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))
# %%
from english_words import get_english_words_set
web2lowerset = get_english_words_set(['gcide'], lower=True)
print(len(web2lowerset), list(web2lowerset)[:10])
valid_tokens_i = [
    # i for i, t in enumerate(all_tok_strs) if t.startswith(" ") and t.lower() == t
    i for i, t in enumerate(all_tok_strs) if t.startswith("▁") and t[1:] in web2lowerset
]
valid_tokens_s = [all_tok_strs[i][1:] for i in valid_tokens_i]
print(len(valid_tokens_i), valid_tokens_s[:10])
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
output = model.lm_head(torch.stack(model(s_ids, output_hidden_states=True).hidden_states))
print(output.shape)
# %%
layer = 20
embeds = output[layer, 0, :, valid_tokens_i]
k = 5
top_k = embeds.topk(k).indices
# %%
for i in range(len(embeds)):
    s = tokenizer.decode(s_ids[0, i]) + " -> " +" ".join([(valid_tokens_s[top_k[i, j]]) for j in range(k)])
    print(s)
# %%
