import math
import torch
import torch.nn as nn

from transformers.pytorch_utils import Conv1D
from transformers.activations import NewGELUActivation

class DotDict(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v
    def __delattr__(self, k): del self[k]


class Attention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, dropout=0.0):
        super().__init__()
        assert embed_size % num_heads == 0
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.c_attn = Conv1D(3 * embed_size, embed_size)
        self.c_proj = Conv1D(embed_size, embed_size)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, E = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.c_attn(x)  # (B, T, 3E)
        q, k, v = qkv.split(E, dim=-1)  # (B, T, E) x3
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)

        # (B, H, T, T)
        att = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(D)
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-1e20"))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # (B, T, H, D) -> (B, T, E)
        y = torch.einsum("bhqk,bkhd->bqhd", att, v).contiguous().view(B, T, E)
        y = self.c_proj(y)
        y = self.resid_drop(y)
        return y

class MLP(nn.Module):
    def __init__(self, embed_size: int, dropout: float):
        super().__init__()
        self.c_fc = Conv1D(4 * embed_size,embed_size)
        self.c_proj = Conv1D(embed_size, 4 * embed_size)
        self.dropout = nn.Dropout(dropout)
        self.act = NewGELUActivation()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class GPT2Block(nn.Module):
    def __init__(self, embed_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = Attention(embed_size, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_size)
        self.mlp  = MLP(embed_size, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln_1(x), mask=mask)  # pre-LN
        x = x + self.mlp(self.ln_2(x))
        return x

def build_causal_mask(B, T, device):
    return torch.ones(T, T, device=device).tril().unsqueeze(0).unsqueeze(1).expand(B, 1, T, T)


class GPT2(nn.Module):
    def __init__(self, vocab_size=50257, max_length=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.1,
                 layer_id=0, n_block=12, reduce_comm=False, bottleneck_dim=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.layer_id = layer_id

        self.config = DotDict(
            model_type="gpt2",
            vocab_size=vocab_size,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            max_position_embeddings=max_length,
            bos_token_id=50256, eos_token_id=50256, pad_token_id=50256,
            is_encoder_decoder=False, tie_word_embeddings=False,
        )

        self.reduce_comm = reduce_comm

        if self.layer_id == 1:
            self.wte = nn.Embedding(vocab_size, n_embd)
            self.wpe = nn.Embedding(max_length, n_embd)
            self.drop = nn.Dropout(dropout)

            self.h = nn.ModuleList([
                GPT2Block(n_embd, n_head, dropout)
                for _ in range(n_block)
            ])

            if self.reduce_comm:
                self.encoder = nn.Sequential(
                    nn.Linear(n_embd, n_embd // 2),
                    nn.LayerNorm(n_embd // 2),
                    nn.GELU(),

                    nn.Linear(n_embd // 2, bottleneck_dim),
                    nn.LayerNorm(bottleneck_dim),
                    nn.GELU(),
                )

        elif self.layer_id == 2:

            if self.reduce_comm:
                self.decoder = nn.Sequential(
                    nn.Linear(bottleneck_dim, n_embd // 2),
                    nn.LayerNorm(n_embd // 2),
                    nn.GELU(),

                    nn.Linear(n_embd // 2, n_embd),
                    nn.LayerNorm(n_embd),
                )

            self.h = nn.ModuleList([
                GPT2Block(n_embd, n_head, dropout)
                for _ in range(n_block)
            ])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        else:
            self.wte = nn.Embedding(vocab_size, n_embd)
            self.wpe = nn.Embedding(max_length, n_embd)
            self.drop = nn.Dropout(dropout)

            self.h = nn.ModuleList([
                GPT2Block(n_embd, n_head, dropout)
                for _ in range(n_block)
            ])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None,
            **kwargs):
        if self.layer_id == 1:
            B, T = input_ids.shape
            h = self.wte(input_ids)
            pos = torch.arange(0, T).unsqueeze(0).to(attention_mask.device)
            h = self.drop(h + self.wpe(pos))
            mask = build_causal_mask(B, T, attention_mask.device)
            if attention_mask is not None:
                # key_mask = attention_mask[:, None, None, :].to(mask.dtype)
                # mask = mask * key_mask
                key_mask = attention_mask[:, None, None, :].to(mask.dtype)
                qry_mask = attention_mask[:, None, :, None].to(mask.dtype)
                mask = mask * key_mask * qry_mask
            for blk in self.h:
                h = blk(h, mask)

            if self.reduce_comm:
                h = self.encoder(h)

        elif self.layer_id == 2:
            h = input_ids
            mask = attention_mask
            if self.reduce_comm:
                h = self.decoder(h)
            for blk in self.h:
                h = blk(h, attention_mask)
            h = self.ln_f(h)
            h = self.lm_head(h)

        else:
            B, T = input_ids.shape
            h = self.wte(input_ids)
            pos = torch.arange(0, T).unsqueeze(0).to(attention_mask.device)
            h = self.drop(h + self.wpe(pos))
            mask = build_causal_mask(B, T, attention_mask.device)
            if attention_mask is not None:
                # key_mask = attention_mask[:, None, None, :].to(mask.dtype)
                # mask = mask * key_mask
                key_mask = attention_mask[:, None, None, :].to(mask.dtype)
                qry_mask = attention_mask[:, None, :, None].to(mask.dtype)
                mask = mask * key_mask * qry_mask

            for blk in self.h:
                h = blk(h, mask)
            h = self.ln_f(h)
            h = self.lm_head(h)

        return h, mask

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        B, T   = input_ids.shape
        device = input_ids.device
        causal = (torch.ones(T, T, device=device)
                  .tril().unsqueeze(0).unsqueeze(1).expand(B, 1, T, T))
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(causal.dtype)
            qry_mask = attention_mask[:, None, :, None].to(causal.dtype)
            mask = causal * key_mask * qry_mask
        else:
            mask = causal
        return {"input_ids": input_ids, "mask": mask}

def shift_labels_for_lm(input_ids, ignore_index=-100):
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = ignore_index
    return labels
