"""
test.py — Evaluate GPT2 checkpoints sau khi train SplitFedLLM

Usage:
    python test.py                        # test GPT2.pt
    python test.py --model GPT2_round3.pt
    python test.py --model GPT2.pt --device cpu
"""
import argparse
import os
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from src.model.GPT2 import GPT2

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="GPT2.pt",  help="Path to .pt checkpoint")
parser.add_argument("--device", default=None,        help="cpu / cuda (auto-detect nếu bỏ qua)")
parser.add_argument("--tokens", default=80, type=int, help="Max new tokens khi generate")
args = parser.parse_args()

DEVICE = torch.device(
    args.device if args.device
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"[INFO] Device: {DEVICE}")

TEST_PROMPTS = [
    "<MR> name[The Golden Curry], food[Fast food], customer rating[low], area[riverside], familyFriendly[yes], near[Café Rouge] </MR> <TEXT>",
    "<MR> name[Fitzbillies], eatType[coffee shop], food[French], priceRange[£20-25], customer rating[3 out of 5] </MR> <TEXT>",
    "<MR> name[The Twenty Two], eatType[restaurant], food[Italian], familyFriendly[no] </MR> <TEXT>",
    "<MR> name[Cotto], eatType[coffee shop], food[Indian], priceRange[moderate], area[riverside], near[The Portland Arms] </MR> <TEXT>",
    "<MR> name[Giraffe], eatType[pub], food[Fast food], area[city centre], familyFriendly[no] </MR> <TEXT>",
]

EVAL_PAIRS = [
    (
        "<MR> name[The Golden Curry], food[Fast food] </MR> <TEXT>",
        "The Golden Curry is a fast food place with a low customer rating located near Café Rouge."
    ),
    (
        "<MR> name[Fitzbillies], eatType[coffee shop], food[French] </MR> <TEXT>",
        "Fitzbillies is a French coffee shop with a customer rating of 3 out of 5."
    ),
]

# ── Load model ────────────────────────────────────────────────────────────────
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy: {path}")

    print(f"\n[INFO] Loading {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    sd = torch.load(path, map_location="cpu")

    model = GPT2()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[DEBUG] Missing keys   : {len(missing)}")
    print(f"[DEBUG] Unexpected keys: {len(unexpected)}")

    if hasattr(model, "lm_head") and hasattr(model, "wte"):
        model.lm_head.weight = model.wte.weight
        print("[OK] Weight tying applied")

    model.to(DEVICE)
    model.eval()
    return model

# ── Generate ──────────────────────────────────────────────────────────────────
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    """Top-k sampling + repetition penalty."""
    ids        = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    prompt_len = ids.shape[1]
    generated  = []

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out    = model(input_ids=ids)
            logits = out["logits"][:, -1, :].clone()  # (1, vocab)

        # Repetition penalty mạnh hơn (1.5 thay vì 1.3)
        for tid in set(ids[0].tolist()):
            logits[0, tid] /= 1.5

        # Penalty riêng cho các từ lặp gần đây (context 20 tokens cuối)
        recent = ids[0, -20:].tolist()
        for tid in set(recent):
            logits[0, tid] /= 1.3

        # Temperature + top-k (top-k=40 để output tập trung hơn)
        logits /= 0.7
        topk_vals, topk_idx = torch.topk(logits, 40)
        probs      = torch.softmax(topk_vals, dim=-1)
        next_token = topk_idx[0, torch.multinomial(probs, 1)]

        if next_token.item() == tokenizer.eos_token_id:
            break

        generated.append(next_token.item())
        ids = torch.cat([ids, next_token.view(1, 1)], dim=1)

    output = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return output

# ── Loss / Perplexity ─────────────────────────────────────────────────────────
def compute_loss(model, tokenizer, pairs: list) -> tuple:
    """
    pairs: list of (prompt, reference) tuples.
    Tính loss chỉ trên phần reference (không tính prompt), giống notebook.
    """
    criterion   = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_loss  = 0.0
    total_tokens = 0

    with torch.no_grad():
        for prompt, ref in pairs:
            full_text  = prompt + " " + ref
            full_ids   = tokenizer.encode(full_text, return_tensors="pt").to(DEVICE)
            prompt_len = len(tokenizer.encode(prompt))

            out    = model(input_ids=full_ids)
            logits = out["logits"]  # (1, seq, vocab)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = full_ids[:, 1:].contiguous().clone()

            # Mask phần prompt: chỉ tính loss trên phần reference
            shift_labels[:, :prompt_len - 1] = -100

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            n_ref_tokens = (shift_labels != -100).sum().item()
            total_loss   += loss.item()
            total_tokens += n_ref_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl      = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl

# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check(model, tokenizer):
    print("\n[Sanity check]")
    ids = tokenizer.encode("The restaurant", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(input_ids=ids)
    print("[DEBUG] Output keys:", list(out.keys()))
    if "logits" not in out:
        raise RuntimeError("Model không trả về 'logits' → kiểm tra GPT2.forward()")
    topk = torch.topk(out["logits"][0, -1], 5)
    print("[DEBUG] Top next tokens:", [tokenizer.decode([i]) for i in topk.indices.tolist()])

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model)
    sanity_check(model, tokenizer)

    # ── Generation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    for prompt in TEST_PROMPTS:
        output = generate(model, tokenizer, prompt, max_new_tokens=args.tokens)
        print(f"\nPROMPT:\n{prompt}")
        print(f"OUTPUT:\n{output}")
        print("-" * 60)

    # ── Loss / Perplexity ─────────────────────────────────────────────────────
    loss, ppl = compute_loss(model, tokenizer, EVAL_PAIRS)
    print("\n" + "=" * 60)
    print(f"[METRIC] Loss: {loss:.4f} | Perplexity: {ppl:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    main()