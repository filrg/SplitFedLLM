import torch
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from transformers import GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from src.model.GPT2 import GPT2

def simple_generate(model, input_ids, attention_mask, max_new_tokens, pad_id, device):
    curr_input = input_ids.to(device)
    curr_mask = attention_mask.to(device)

    for _ in range(max_new_tokens):
        logits, _ = model(curr_input, attention_mask=curr_mask)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        curr_input = torch.cat([curr_input, next_token], dim=1)
        curr_mask = torch.cat([curr_mask, torch.ones((1, 1), device=input_ids.device)], dim=1)

        if next_token.item() == pad_id:
            break

    return curr_input

def compute_bleu(references, predictions):

    smooth = SmoothingFunction().method1
    scores = []

    for ref, pred in zip(references, predictions):

        ref_tokens = [r.split() for r in ref]
        pred_tokens = pred.split()

        score = sentence_bleu(
            ref_tokens,
            pred_tokens,
            smoothing_function=smooth
        )
        scores.append(score)

    return sum(scores) / len(scores)

def compute_rouge_l(references, predictions):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    scores = []

    for refs, pred in zip(references, predictions):
        best_score = 0

        for ref in refs:
            score = scorer.score(ref, pred)
            rouge_l = score['rougeL'].fmeasure
            best_score = max(best_score, rouge_l)

        scores.append(best_score)

    return sum(scores) / len(scores)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
pad_id = tokenizer.eos_token_id
model = GPT2()

state_dict = torch.load("GPT2.pt", map_location=device)
model.load_state_dict(state_dict)
model.to(device)

test_samples = pd.read_csv("./data/E2E/devset.csv")

test_data = []

for _, row in test_samples.iterrows():
    test_data.append({
        "mr": row["mr"],
        "ref": row["ref"]
    })

mr_to_refs = defaultdict(list)

for item in test_data:
    mr = item["mr"]
    ref = item["ref"]
    mr_to_refs[mr].append(ref)

references, predictions = [], []

for mr, ref in tqdm(mr_to_refs.items()):

    prompt = f"<MR> {mr} </MR> Answer: "
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)

    output_ids = simple_generate(model, input_ids, attention_mask, 50, pad_id, device)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    answer = full_text.split("Answer: ")[-1].strip()

    predictions.append(answer)
    references.append(ref)
    break

# bleu = compute_bleu(references, predictions)
rouge_l = compute_rouge_l(references, predictions)

# print(bleu)
print(rouge_l)



