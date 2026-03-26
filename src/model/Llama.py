import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from src.Optimizer import flash_scaled_dot_product

class DotDict(dict):
    def __getattr__(self, k):
        try: return self[k]
        except KeyError: raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v
    def __delattr__(self, k): del self[k]

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def build_causal_mask(B, T, device):
    return torch.ones(T, T, device=device).tril().unsqueeze(0).unsqueeze(1).expand(B, 1, T, T).contiguous()

class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

        self._set_cos_sin_cache(max_position_embeddings, dtype=torch.float32, device=device)

    def _set_cos_sin_cache(self, seq_len: int, dtype: torch.dtype, device=None):
        # Fix Bug 12: khi device=None (gọi lại từ forward), dùng inv_freq.device
        device = device if device is not None else self.inv_freq.device

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freq = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freq, freq], dim=-1)

        cos = emb.cos()[None, None, :, :]  # [1, 1, seq_len, dim]
        sin = emb.sin()[None, None, :, :]

        # Lưu FP32 rồi cast khi dùng để tránh precision mismatch
        self.cos_cached = cos.to(dtype=torch.float32, device=device)
        self.sin_cached = sin.to(dtype=torch.float32, device=device)
        self.max_seq_len_cached = seq_len

    def forward(self, x, seq_len: int = None):
        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, dtype=torch.float32, device=x.device)

        # Cast về dtype và device của x tại thời điểm dùng (hỗ trợ FP16/BF16)
        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device)
        return cos, sin

class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, dropout=0.0, head_dim=64,
                 use_flash=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.use_flash = use_flash

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.rope = LlamaRotaryEmbedding(head_dim)
        self.dropout = nn.Dropout(dropout)
        self._dropout_p = dropout

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

        dp = self._dropout_p if self.training else 0.0
        out = flash_scaled_dot_product(q, k, v, mask=attention_mask, dropout_p=dp)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, act_fn=F.silu):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = act_fn

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size,
                 rms_eps=1e-6, use_flash=False):
        super().__init__()
        self.input_layernorm  = LlamaRMSNorm(hidden_size, eps=rms_eps)
        self.self_attn = LlamaAttention(hidden_size, num_heads, num_kv_heads,
                                        use_flash=use_flash)
        self.post_attention_layernorm  = LlamaRMSNorm(hidden_size, eps=rms_eps)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def forward(self, x, attention_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Llama(nn.Module):
    def __init__(self, vocab_size=32000, hidden_size=768, intermediate_size=3072, num_attention_heads=12,
                 num_key_value_heads=12,
                 layer_id=0, n_block=12, use_flash=False):
        super().__init__()
        self.layer_id = layer_id
        self.use_flash = use_flash
        self.config = DotDict(
            model_type="llama",
            vocab_size=vocab_size,
            max_position_embeddings=2048,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            initializer_range=0.02,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=1,
            is_encoder_decoder=False,
            tie_word_embeddings=False,
            use_cache=True,
            torch_dtype="float32",
        )

        def _make_layers(n):
            return nn.ModuleList([
                LlamaDecoderLayer(hidden_size, num_attention_heads,
                                  num_key_value_heads, intermediate_size,
                                  use_flash=use_flash)
                for _ in range(n)
            ])

        if self.layer_id == 1:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = _make_layers(n_block)
        elif self.layer_id == 2:
            self.layers = _make_layers(n_block)
            self.norm = LlamaRMSNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.layers = _make_layers(n_block)
            self.norm = LlamaRMSNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if self.layer_id == 1:
            B, T = input_ids.shape
            x = self.embed_tokens(input_ids)

            # Fix: dùng input_ids.device làm fallback khi attention_mask là None
            ref_device = attention_mask.device if attention_mask is not None else input_ids.device
            masks = build_causal_mask(B, T, ref_device)
            if attention_mask is not None:
                key_mask = attention_mask[:, None, None, :].to(masks.dtype)
                qry_mask = attention_mask[:, None, :, None].to(masks.dtype)
                masks = masks * key_mask * qry_mask

            for decode in self.layers:
                x = decode(x, masks)

        elif self.layer_id == 2:
            x = input_ids
            masks = attention_mask
            for decode in self.layers:
                x = decode(x, masks)
            x = self.norm(x)
            x = self.lm_head(x)

        else:
            B, T = input_ids.shape
            x = self.embed_tokens(input_ids)

            ref_device = attention_mask.device if attention_mask is not None else input_ids.device
            masks = build_causal_mask(B, T, ref_device)
            if attention_mask is not None:
                key_mask = attention_mask[:, None, None, :].to(masks.dtype)
                qry_mask = attention_mask[:, None, :, None].to(masks.dtype)
                masks = masks * key_mask * qry_mask

            for decode in self.layers:
                x = decode(x, masks)

            x = self.norm(x)
            x = self.lm_head(x)

        return x, masks

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        B, T = input_ids.shape
        device = input_ids.device
        causal = torch.ones(T, T, device=device).tril().unsqueeze(0).unsqueeze(1).expand(B, 1, T, T).contiguous()
