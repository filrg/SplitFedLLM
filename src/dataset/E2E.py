import torch
from torch.utils.data import Dataset

class E2E_DATASET(Dataset):
    def __init__(self, tokenizer, dataset, max_length=128):
        self.pad_id = tokenizer.eos_token_id
        self.max_length = max_length
        self.samples = []

        for data in dataset:
            prompt = f"<MR> {data['mr']} </MR> Answer: "
            target = data["ref"] + tokenizer.eos_token

            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(target, add_special_tokens=False)

            input_ids = prompt_ids + target_ids

            prompt_len = len(prompt_ids)
            input_len = len(input_ids)
            if input_len < self.max_length:
                pad_len = self.max_length - input_len
            else:
                continue

            attention_mask = [1] * input_len + [0] * pad_len
            input_ids_pad = input_ids + [self.pad_id] * pad_len

            labels = ([-100] * prompt_len
                      + target_ids
                      + [-100] * pad_len)

            self.samples.append({
                "input_ids": torch.tensor(input_ids_pad, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

