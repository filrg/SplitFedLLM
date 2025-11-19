import torch

class CodeAlpacaDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, set_max_len=1024, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["prompt"] for ex in self.examples]
        self.ans = [ex["completion"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix

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
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
        )

        if len(tokens) > self.max_len:
            tokens, mask = tokens[:self.max_len], mask[:self.max_len]
        elif len(tokens) <= self.max_len:
            pad_len = self.max_len - len(tokens)
            tokens = tokens + [0] * pad_len
            mask = mask + [0] * pad_len

        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask)


# def get_examples(split, eos_token="<|endoftext|>"):
#     def format_example(example):
#         example['prompt'] = example['prompt'] + "\n"
#         example['completion'] = example['completion'] + eos_token
#         return example
#     transformed_dataset = split.map(format_example)
#
#     examples_list = list(transformed_dataset)
#     print(f"{len(examples_list)} {split} examples")
#     return examples_list