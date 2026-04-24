from torch.utils.data import Dataset
import torch

class SQUAD_DATASET(Dataset):
    def __init__(self, tokenizer , examples: list, max_length: int = 256):
        self.pad_id     = tokenizer.eos_token_id
        self.max_length = max_length
        self.samples    = []

        for ex in examples:
            context  = ex["context"]
            question = ex["question"]
            answer   = ex["answers"][0]

            prefix = f"Context: {context}\nQuestion: {question}\nAnswer: "
            suffix = answer + tokenizer.eos_token

            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

            if len(prefix_ids) + len(suffix_ids) > max_length:
                keep = max_length - len(suffix_ids)
                prefix_ids = prefix_ids[-keep:] if keep > 0 else []

            input_ids  = prefix_ids + suffix_ids
            prefix_len = len(prefix_ids)
            seq_len    = len(input_ids)

            pad_len        = max_length - seq_len
            attention_mask = [1] * seq_len + [0] * pad_len
            input_ids_pad  = input_ids + [self.pad_id] * pad_len
            labels         = ([-100] * prefix_len
                               + suffix_ids
                               + [-100] * pad_len)

            self.samples.append({
                "input_ids":      torch.tensor(input_ids_pad,   dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask,  dtype=torch.long),
                "labels":         torch.tensor(labels,          dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]