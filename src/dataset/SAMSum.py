import torch

class SAMSumDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, set_max_len=1024, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["dialogue"] for ex in self.examples]
        self.ans = [ex["summary"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.pad_id = tokenizer.pad_token_if if tokenizer.pad_token_id else tokenizer.eos_token_id
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        self.max_len = min(self.max_len, set_max_len)
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]

        tokens = qn_tokens + ans_tokens
        mask = [1] * len(tokens)

        pad_len = self.max_len - len(tokens)
        if pad_len >= 0:
            tokens += [self.pad_id] * pad_len
            mask += [0] * pad_len
        else:
            tokens, mask = tokens[:self.max_len], mask[self.max_len]

        labels = tokens.copy()
        if not self.loss_on_prefix:
            for i in range(len(qn_tokens)):
                labels[i] = self.pad_id

        tokens = torch.tensor(tokens, dtype=torch.long)
        mask = torch.tensor(mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        return dict(
            input_ids=tokens,
            attention_mask=mask,
            labels=labels
        )