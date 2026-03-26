import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.dataset.dataloader import dataloader
from transformers import GPT2Tokenizer
from src.model.GPT2 import GPT2

import re

# Metric cho E2E (cần cài: pip install sacrebleu rouge-score)
try:
    from sacrebleu import corpus_bleu
    from rouge_score import rouge_scorer as rouge_scorer_lib
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False


def extract_final_number(s: str) -> str:
    if s is None:
        return ""
    m = re.search(r"####\s*([\-+]?\d+(?:\.\d+)?)", s)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[\-+]?\d+(?:\.\d+)?", s)
    if nums:
        return nums[-1].strip()
    return ""


def _greedy_generate(model, prompt_ids, prompt_mask, max_new_tokens, pad_id, device):
    cur_ids  = prompt_ids.clone()
    cur_mask = prompt_mask.clone()
    generated = []

    for _ in range(max_new_tokens):
        out        = model(input_ids=cur_ids, attention_mask=cur_mask)
        logits     = out["hidden_states"]
        next_token = logits[0, -1, :].argmax(-1).item()
        generated.append(next_token)

        if next_token == pad_id:
            break

        next_tensor = torch.tensor([[next_token]], device=device)
        next_mask   = torch.ones((1, 1), device=device, dtype=cur_mask.dtype)
        cur_ids  = torch.cat([cur_ids,  next_tensor], dim=1)
        cur_mask = torch.cat([cur_mask, next_mask],   dim=1)

    return generated


def val_GPT2(model_name, data_name, state_dict_full, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval device:", device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')

    test_loader = dataloader(model_name=model_name, data_name=data_name, train=False)

    model = GPT2()

    pretrained_path = f"{model_name}.pt"
    if os.path.exists(pretrained_path):
        base_state = torch.load(pretrained_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(base_state, strict=False)
        if missing:
            logger.log_info(f"[val_GPT2] Base load missing keys: {missing}")
        if unexpected:
            logger.log_info(f"[val_GPT2] Base load unexpected keys: {unexpected}")
    else:
        logger.log_warning(
            f"[val_GPT2] Pretrained file '{pretrained_path}' not found. "
            f"Validating with random base weights."
        )

    if state_dict_full:
        missing, unexpected = model.load_state_dict(state_dict_full, strict=False)
        if missing:
            logger.log_info(f"[val_GPT2] LoRA overlay missing keys: {missing}")
        if unexpected:
            logger.log_info(f"[val_GPT2] LoRA overlay unexpected keys: {unexpected}")

    model = model.to(device)
    model.eval()

    total_loss      = 0.0
    total_tokens    = 0
    total_samples   = 0
    correct_samples = 0

    nl_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    is_e2e = (data_name == 'E2E')
    hypotheses = []
    references = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            mask      = batch['attention_mask'].to(device)
            labels    = batch['labels'].to(device)

            out    = model(input_ids, mask)
            logits = out["hidden_states"]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            valid         = (shift_labels != pad_id).sum().item()
            total_loss   += loss.item()
            total_tokens += valid

            for i in range(input_ids.size(0)):
                tokens = input_ids[i].tolist()
                if nl_id in tokens:
                    cut = tokens.index(nl_id) + 1
                else:
                    cut = len(tokens) // 2

                prompt_ids  = input_ids[i, :cut].unsqueeze(0)
                prompt_mask = mask[i, :cut].unsqueeze(0)

                generated = _greedy_generate(
                    model, prompt_ids, prompt_mask,
                    max_new_tokens=64, pad_id=pad_id, device=device
                )

                gen_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
                ref_text = tokenizer.decode(
                    labels[i][labels[i] != pad_id], skip_special_tokens=True
                ).strip()

                total_samples += 1

                if is_e2e:
                    # E2E: dùng BLEU/ROUGE
                    hypotheses.append(gen_text)
                    references.append(ref_text)
                else:
                    # GSM8K: dùng exact match số
                    pred_num = extract_final_number(gen_text)
                    gold_num = extract_final_number(ref_text)
                    if gold_num and pred_num == gold_num:
                        correct_samples += 1

    avg_loss = total_loss / max(total_tokens, 1)

    if is_e2e and BLEU_AVAILABLE and hypotheses:
        bleu = corpus_bleu(hypotheses, [references]).score
        scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_l = sum(
            scorer.score(r, h)["rougeL"].fmeasure
            for h, r in zip(hypotheses, references)
        ) / len(hypotheses)
        print(f"Loss / token : {avg_loss:.4f}; BLEU: {bleu:.2f}; ROUGE-L: {rouge_l:.4f}")
        logger.log_info(f"Loss / token : {avg_loss:.4f}; BLEU: {bleu:.2f}; ROUGE-L: {rouge_l:.4f}")
    else:
        accuracy = correct_samples / max(total_samples, 1)
        print(f"Loss / token : {avg_loss:.4f}; Accuracy (answer) : {accuracy * 100:.2f}%")
        logger.log_info(f"Loss / token : {avg_loss:.4f}; Accuracy (answer) : {accuracy * 100:.2f}%")

    return model.state_dict()