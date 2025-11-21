import re
import torch

from transformers import GPT2Tokenizer

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSM8K(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            pad_id = tokenizer.eos_token_id
        self.pad_id = pad_id

        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)

        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]

        tokens = qn_tokens + ans_tokens

        attn = [1] * len(tokens)

        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens += [self.pad_id] * pad_len
            attn += [0] * pad_len

        labels = tokens.copy()

        if not self.loss_on_prefix:
            q_len = len(qn_tokens)
            for i in range(q_len):
                labels[i] = self.pad_id

        tokens = torch.tensor(tokens, dtype=torch.long)
        attn = torch.tensor(attn, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return dict(
            input_ids=tokens,
            attention_mask=attn,
            labels=labels,
        )
