import re
import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from src.dataset.dataloader import dataloader
from src.model.GPT2 import GPT2


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

def val_GPT2(model_name, data_name, state_dict_full, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval device:", device)

    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    pad_id = tokenizer.pad_token_id

    loss_fct = nn.CrossEntropyLoss(
        ignore_index=pad_id,
    )

    test_loader = dataloader(model_name=model_name, data_name=data_name, train=False)

    model = GPT2()
    model.load_state_dict(state_dict_full)
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    total_bleu = 0.0
    total_rouge1 = 0.0
    total_rouge2 = 0.0
    total_rougeL = 0.0

    total_samples = 0
    correct_samples = 0

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
                shift_labels.view(-1)
            )
            total_loss += loss.item()

            predicted_tokens = torch.argmax(logits, dim=-1)

            generated_texts = [
                tokenizer.decode(p, skip_special_tokens=True).strip()
                for p in predicted_tokens
            ]
            reference_texts = [
                tokenizer.decode(l, skip_special_tokens=True).strip()
                for l in labels
            ]

            for gen, ref in zip(generated_texts, reference_texts):
                if not gen: gen = " "
                if not ref: ref = " "
            #     print(f'gen : {gen}')
            #     print(f'ref : {ref}')
            #     break
            # break
                if data_name == 'GSM8K':
                    pred_num = extract_final_number(gen)
                    gold_num = extract_final_number(ref)
                    if gold_num and pred_num == gold_num:
                        correct_samples += 1

                bleu_score = sentence_bleu(
                    [ref.split()],
                    gen.split(),
                    smoothing_function=smooth
                )
                total_bleu += bleu_score

                try:
                    scores = scorer.score(ref, gen)
                    total_rouge1 += scores['rouge1'].fmeasure
                    total_rouge2 += scores['rouge2'].fmeasure
                    total_rougeL += scores['rougeL'].fmeasure
                except Exception as e:
                    print(f"Exception in val_GPT2: {e}")

                total_samples += 1
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = correct_samples / max(total_samples, 1)

    avg_bleu = total_bleu / max(total_samples, 1)
    avg_rouge1 = total_rouge1 / max(total_samples, 1)
    avg_rouge2 = total_rouge2 / max(total_samples, 1)
    avg_rougeL = total_rougeL / max(total_samples, 1)

    print(f"Loss / token : {avg_loss:.4f}")
    if data_name == 'GSM8K':
        print(f"; Accuracy (answer): {accuracy * 100:.2f}%")
    print(f"; BLUE: {avg_bleu:.4f}; Rouge-1: {avg_rouge1:.4f}; Rouge-2: {avg_rouge2:.4f}; Rouge-L: {avg_rougeL:.4f}")

    logger.log_info(
        f"Loss / token : {avg_loss:.4f}; Accuracy (answer) : {accuracy * 100:.2f}%"
        f"; BLUE: {avg_bleu:.4f}; Rouge-1: {avg_rouge1:.4f}; Rouge-2: {avg_rouge2:.4f}; Rouge-L: {avg_rougeL:.4f}"
    )
