import torch
import torch.nn as nn
from tqdm import tqdm

# import src.Log
from src.dataset.dataloader import dataloader
from transformers import GPT2Tokenizer
from src.model.GPT2 import GPT2

import re
import string
import unicodedata

from collections import Counter

def simple_generate(model, input_ids, attention_mask, max_new_tokens, eos_token_id, device):
    curr_input = input_ids.to(device)
    curr_mask = attention_mask.to(device)
    for _ in range(max_new_tokens):
        logits, _ = model(curr_input, attention_mask=curr_mask)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        curr_input = torch.cat([curr_input, next_token], dim=1)
        curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=input_ids.device)], dim=1)
        if next_token.item() == eos_token_id:
            break
    return curr_input

def normalize_answer(s):
    s = s.lower()

    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')

    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ' '.join(s.split())
    return s

def compute_f1(pred, gt):
    pred_tokens = normalize_answer(pred).split()
    gt_tokens   = normalize_answer(gt).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    if precision + recall == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 4)

def val_GPT2(state_dict_full, cut_layers , bottleneck_config, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval device:", device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    pad_id = tokenizer.eos_token_id

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    test_loader = dataloader(model_name='GPT2', batch_size=2, distribution=[1000], train=False)

    model = GPT2()
    model.load_state_dict(state_dict_full)
    model = model.to(device)

    model.eval()

    total_loss, total_f1, total_em , n = 0.0, 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits, _ = model(input_ids, mask)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            if torch.isnan(loss).any():
                print("NaN detected in loss")
                return False

            total_loss   += loss.item()

        test_data = torch.load("./data/SQUAD/VAL_SQUAD.pt")
        test_samples = test_data[:500]

        for sample in tqdm(test_samples):
            context = sample['context']
            question = sample['question']
            ground_truth = sample['answers'][0]

            prompt = f"Context: {context}\nQuestion: {question}\nAnswer: "
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            attention_mask = torch.ones(input_ids.shape, device=device)

            output_ids = simple_generate(model, input_ids, attention_mask, 20, pad_id, device)
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer = full_text.split("Answer: ")[-1].strip()
            answer = answer.split("\n")[0]

            if answer.strip().lower() == ground_truth.strip().lower():
                total_em += 1

            f1 = compute_f1(answer, ground_truth)

            total_f1 += f1
            n += 1

    avg_loss = total_loss / max(len(test_loader), 1)
    avg_f1 = total_f1 / max(n, 1) * 100
    total_em = total_em / max(n, 1) * 100

    print(f"Test Loss: {avg_loss:.4f}; F1: {avg_f1:.4f}; EM: {total_em:.4f}")
    logger.log_info(f"Test Loss: {avg_loss:.4f}; F1: {avg_f1:.4f}; EM: {total_em:.4f}")

    return True