import torch
from transformers import GPT2LMHeadModel

# load HF model
model = GPT2LMHeadModel.from_pretrained("./gpt2_e2e_finetuned")

# lấy state_dict
state_dict = model.state_dict()

# save về format server dùng
torch.save(state_dict, "GPT2.pt")

print("Saved GPT2.pt")