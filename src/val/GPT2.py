import torch
import torch.nn as nn
from tqdm import tqdm

# import src.Log
from src.dataset.dataloader import dataloader
from transformers import GPT2Tokenizer
from src.model.GPT2 import GPT2

import re
import string
from collections import Counter

def normalize_answer(s):
    s = s.lower()
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

    return 2 * precision * recall / (precision + recall)

def val_GPT2(state_dict_full, cut_layers , bottleneck_config, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval device:", device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    pad_id = tokenizer.eos_token_id

    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    test_loader = dataloader(model_name='GPT2', batch_size=4, distribution=[200], train=False)

    total_loss, total_f1, n = 0.0, 0, 0

    if bottleneck_config['enable']:
        client = GPT2(layer_id=1, n_block=cut_layers)
        client_state_dict = client.state_dict()
        for key in client_state_dict:
            client_state_dict[key] = state_dict_full[0][key]
        client.load_state_dict(client_state_dict)
        client = client.to(device)
        server = GPT2(layer_id=2, n_block=12 - cut_layers)
        server_state_dict = server.state_dict()
        for key in server_state_dict:
            server_state_dict[key] = state_dict_full[1][key]
        server.load_state_dict(server_state_dict)
        server = server.to(device)
    else:
        client = GPT2(layer_id=1, n_block=cut_layers)
        client.load_state_dict(state_dict_full[0])
        client = client.to(device)
        server = GPT2(layer_id=2, n_block= 12 - cut_layers)
        server.load_state_dict(state_dict_full[1])
        server = server.to(device)

    client.eval()
    server.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits, mask = client(input_ids, mask)
            logits, _ = server(logits, mask)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            total_loss   += loss.item()
            predicted_ids = torch.argmax(logits, dim=-1)

            for i in range(predicted_ids.size(0)):
                lbl = labels[i]

                answer_mask = (lbl != pad_id).nonzero(as_tuple=False)
                if len(answer_mask) == 0:
                    continue

                s = answer_mask[0].item()
                e = answer_mask[-1].item() + 1

                pred_text = tokenizer.decode(
                    predicted_ids[i, s:e],
                    skip_special_tokens=True
                )

                gold_text = tokenizer.decode(
                    lbl[s:e],
                    skip_special_tokens=True
                )

                f1 = compute_f1(pred_text, gold_text)

                total_f1 += f1
                n += 1

        avg_loss = total_loss / max(len(test_loader), 1)
        avg_f1 = total_f1 / max(n, 1) * 100

        print(f"Test Loss: {avg_loss:.4f}; F1: {avg_f1:.4f}")
        logger.log_info(f"Test Loss: {avg_loss:.4f}; F1: {avg_f1:.4f}")

    return True