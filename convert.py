import torch
from transformers import GPT2LMHeadModel

# load HF model
hf_model = GPT2LMHeadModel.from_pretrained("./gpt2_e2e_finetuned")

hf_sd = hf_model.state_dict()
new_sd = {}

for k, v in hf_sd.items():
    new_k = k

    # 🔥 remove prefix
    if k.startswith("transformer."):
        new_k = k.replace("transformer.", "")

    # 🔥 lm_head giữ nguyên
    new_sd[new_k] = v

# save
torch.save(new_sd, "GPT2.pt")

print("Converted → GPT2.pt")