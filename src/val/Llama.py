import torch
import torch.nn as nn
from tqdm import tqdm
from src.dataset.dataloader import dataloader
from transformers import AutoTokenizer
from src.model.Llama import Llama
import math

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def val_Llama(model_name, data_name, state_dict_full, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Eval device:", device)

    smooth  = SmoothingFunction().method1
    scorer  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m")
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    test_loader = dataloader(model_name=model_name, data_name=data_name, train=False)

    model = Llama()
    model.load_state_dict(state_dict_full)
    model = model.to(device)
    model.eval()

    # Fix: tạo criterion 1 lần ngoài vòng lặp (tránh tạo object mới mỗi batch)
    loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')

    total_bleu       = 0.0
    total_rouge      = 0.0
    total_log_loss   = 0.0   # Fix: tích lũy log-loss thay vì perplexity trực tiếp
    total_valid_tok  = 0     # Fix: đếm token hợp lệ để avg chính xác
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            mask      = batch['attention_mask'].to(device)
            labels    = batch['input_ids'].to(device)   # causal LM: labels = input

            logits, _ = model(input_ids, mask)

            predicted_tokens = torch.argmax(logits, dim=-1)
            generated_texts  = [
                tokenizer.decode(p, skip_special_tokens=True).strip()
                for p in predicted_tokens
            ]
            reference_texts  = [
                tokenizer.decode(l, skip_special_tokens=True).strip()
                for l in labels
            ]

            for gen, ref in zip(generated_texts, reference_texts):
                gen = gen or " "
                ref = ref or " "

                bleu = sentence_bleu([ref.split()], gen.split(),
                                     smoothing_function=smooth)
                try:
                    rouge_l = scorer.score(ref, gen)["rougeL"].fmeasure
                except Exception:
                    rouge_l = 0.0

                total_bleu  += bleu
                total_rouge += rouge_l
                count += 1

            # Fix: perplexity = exp(avg_cross_entropy_per_token)
            # Tích lũy tổng cross-entropy và tổng token; tính exp 1 lần cuối
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss        = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            valid_tokens     = (shift_labels != pad_id).sum().item()
            total_log_loss  += loss.item()
            total_valid_tok += max(valid_tokens, 1)

    # Fix: tính perplexity từ avg log-loss (tránh exp overflow mỗi batch)
    avg_log_loss  = total_log_loss / max(total_valid_tok, 1)
    # Clamp để tránh math overflow với exp()
    avg_perplexity = math.exp(min(avg_log_loss, 20.0))

    avg_bleu  = total_bleu  / max(count, 1)
    avg_rouge = total_rouge / max(count, 1)

    print(f"Evaluation Results: BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, Perplexity: {avg_perplexity:.4f}")
    logger.log_info(
        f"Evaluation Results: BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, Perplexity: {avg_perplexity:.4f}"
    )
