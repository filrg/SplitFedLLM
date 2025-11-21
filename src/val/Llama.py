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

    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-160m")
    tokenizer.pad_token = tokenizer.eos_token

    test_loader = dataloader(model_name=model_name, data_name=data_name, train=False)

    model = Llama()
    model.load_state_dict(state_dict_full)
    model = model.to(device)
    model.eval()

    total_bleu = 0.0
    total_rouge = 0.0
    total_perplexity = 0.0
    count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['input_ids'].to(device)

            logits, _ = model(input_ids, mask)

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

                bleu_score = sentence_bleu(
                    [ref.split()],
                    gen.split(),
                    smoothing_function=smooth
                )

                try:
                    rouge_l = scorer.score(ref, gen)["rougeL"].fmeasure
                except:
                    rouge_l = 0.0

                total_bleu += bleu_score
                total_rouge += rouge_l
                count += 1

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id,
                reduction='sum'
            )

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            valid_tokens = (shift_labels != tokenizer.pad_token_id).sum().item()
            perplexity = math.exp(loss.item() / max(1, valid_tokens))
            total_perplexity += perplexity

    avg_bleu = total_bleu / count
    avg_rouge = total_rouge / count
    avg_perplexity = total_perplexity / count

    print(f"Evaluation Results: BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, Perplexity: {avg_perplexity:.4f}")

    logger.log_info(
        f"Evaluation Results: BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}, Perplexity: {avg_perplexity:.4f}"
    )
