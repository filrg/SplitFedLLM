

import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager
from transformers.pytorch_utils import Conv1D

try:
    from peft import (
        LoraConfig, TaskType, get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import BitsAndBytesConfig
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import bitsandbytes as bnb
    HAS_BNB_PKG = True
except ImportError:
    HAS_BNB_PKG = False


def detect_fan_in_fan_out(model):
    for module in model.modules():
        if isinstance(module, Conv1D):
            return True
    return False

def build_qlora_config(qlora_cfg: dict,model, model_name: str):

    if not HAS_BNB_PKG or not HAS_PEFT:
        return None, None

    bits = qlora_cfg.get("bits", 4)
    double_quant = qlora_cfg.get("double_quant", True)
    r = qlora_cfg.get("r", 8)
    alpha = qlora_cfg.get("alpha", 16)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float16
),
    )

    target_map = {
        "GPT2":  ["c_attn", "c_proj", "c_fc"],
        "Llama": ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"],
        "Bert":  ["query", "key", "value", "dense"],
    }
    targets = target_map.get(model_name, ["query", "value"])

    task = TaskType.SEQ_CLS if model_name == "Bert" else TaskType.CAUSAL_LM
    fan_in = detect_fan_in_fan_out(model)

    lora_config = LoraConfig(
        task_type=task,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=targets,
        fan_in_fan_out=fan_in,
    )
    return bnb_config, lora_config


def apply_qlora(model, lora_config, model_name: str):

    if not HAS_PEFT:
        return model
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = get_peft_model(model, lora_config)
    # Bert cần classifier trainable
    if model_name == "Bert":
        for param in model.classifier.parameters():
            param.requires_grad = True
    model.print_trainable_parameters()
    return model



import torch.nn.functional as F

def flash_scaled_dot_product(q, k, v, mask=None, dropout_p=0.0):


    if not hasattr(F, "scaled_dot_product_attention"):
        # fallback thủ công (giữ nguyên của bạn)
        import math
        scale = math.sqrt(q.size(-1))
        q_fp32 = q.float()
        k_fp32 = k.float()
        v_fp32 = v.float()
        att = (q_fp32 @ k_fp32.transpose(-2, -1)) / scale
        if mask is not None:
            fill_val = torch.finfo(att.dtype).min / 2
            att = att.masked_fill(mask == 0, fill_val)
        att = torch.softmax(att, dim=-1)
        if dropout_p > 0.0:
            att = F.dropout(att, p=dropout_p, training=True)
        out = att @ v_fp32
        return out.to(q.dtype)

    # 👉 convert mask sang đúng format SDPA
    attn_mask = None
    if mask is not None:
        attn_mask = ~mask.bool()  # SDPA dùng attn_mask với True = masked

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=mask is None
    )

@contextmanager
def precision_context(precision: str, device: str = "cuda"):

    if not torch.cuda.is_available() or precision == "fp32":
        yield
        return

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    if precision in dtype_map:
        with torch.autocast(device_type="cuda", dtype=dtype_map[precision]):
            yield
    elif precision == "int8":

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
    else:
        yield


def cast_model_precision(model: nn.Module, precision: str) -> nn.Module:
    """
    Đổi dtype toàn bộ tham số model (dùng khi không có QLoRA).
    int8 dùng bitsandbytes LinearInt8 thay thế nn.Linear.
    """
    if precision == "fp16":
        return model.half()
    if precision == "bf16":
        return model.to(torch.bfloat16)
    if precision == "int8":
        if HAS_BNB_PKG:
            return _replace_linear_int8(model)
        else:
            print("[Optimizer] bitsandbytes chưa cài")
    return model  # fp32 mặc định


def _replace_linear_int8(model: nn.Module) -> nn.Module:

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            has_bias = module.bias is not None
            new_layer = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=has_bias,
                has_fp16_weights=False,
                threshold=6.0,     # LLM.int8() outlier threshold
            )
            new_layer.weight = nn.Parameter(module.weight.data)
            if has_bias:
                new_layer.bias = nn.Parameter(module.bias.data)
            setattr(model, name, new_layer)
        else:
            _replace_linear_int8(module)
    return model




def quantize_hidden(tensor: torch.Tensor) -> tuple:
    """
    Nén hidden state từ FP16/FP32 → INT8 trước khi pickle + gửi.
    Dùng symmetric per-tensor quantization:
        scale  = max(|x|) / 127
        q      = round(x / scale).clamp(-127, 127).to(int8)

    Trả về (q_numpy: np.ndarray[int8], scale: float)
    Kích thước giảm ~4x so với float32, ~2x so với float16.
    """
    t = tensor.detach().float()
    scale = t.abs().max().item() / 127.0
    if scale == 0.0:
        scale = 1e-8
    q = (t / scale).round().clamp(-127, 127).to(torch.int8)
    return q.cpu().numpy(), scale


def dequantize_hidden(q_numpy: np.ndarray, scale: float,
                      device: str, requires_grad: bool = False) -> torch.Tensor:
    """
    Giải nén INT8 → FP32 / FP16 rồi đẩy lên đúng device.
    """
    q = torch.from_numpy(q_numpy).float() * scale
    q = q.to(device)
    if requires_grad:
        q.requires_grad_(True)
    return q



def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Bật gradient checkpointing cho bất kỳ model nào có method
    gradient_checkpointing_enable() (HuggingFace convention),
    hoặc tự áp dụng qua torch.utils.checkpoint.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[Optimizer] gradient_checkpointing_enable() called.")
    elif hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        # Wrap từng TransformerBlock / BertLayer / DecoderLayer
        _wrap_checkpointing(model)
    return model


def _wrap_checkpointing(model: nn.Module):
    """
    Với custom model không phải HuggingFace, tìm và wrap các block
    có tên chứa 'layer', 'block', 'decoder'.
    """
    from torch.utils.checkpoint import checkpoint as ckpt

    KEYWORDS = ("layer", "block", "decoder")

    for name, module in model.named_children():
        if any(k in name.lower() for k in KEYWORDS):
            orig_forward = module.forward

            def make_checkpointed(fwd=orig_forward):
                def checkpointed_forward(*args, **kwargs):
                    def run(*a, **kw):
                        return fwd(*a, **kw)
                    return ckpt(run, *args, **kwargs, use_reentrant=False)
                return checkpointed_forward

            module.forward = make_checkpointed()
        else:
            _wrap_checkpointing(module)



def make_scaler(precision: str):
    """
    Trả về GradScaler nếu precision == 'fp16', ngược lại None.
    Dùng cùng optimizer.step():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    """
    if precision == "fp16" and torch.cuda.is_available():
        return torch.amp.GradScaler("cuda")
    return None


class OptimizationBundle:
    """
    Gói tất cả cấu hình tối ưu vào một object duy nhất để
    truyền qua RpcClient → Ft_* một cách gọn gàng.
    """
    def __init__(self, opt_cfg: dict):
        
        self.precision = opt_cfg.get("precision", "fp32")
        self.quantize_hidden = opt_cfg.get("quantize_hidden", False)
        self.gradient_checkpointing = opt_cfg.get("gradient_checkpointing", False)
        self.scaler = make_scaler(self.precision)
        self.flash_attention = opt_cfg.get("flash_attention", False)
    def precision_ctx(self, device="cuda"):
        return precision_context(self.precision, device)

    def quant(self, tensor: torch.Tensor):
        """Nén hidden state nếu được bật."""
        if self.quantize_hidden:
            return quantize_hidden(tensor)
        # Trả về (numpy fp16, scale=None) để interface nhất quán
        return tensor.detach().cpu().to(torch.float16).numpy(), None

    def dequant(self, q_numpy, scale, device, requires_grad=False):
        """Giải nén hidden state."""
        if scale is not None:
            return dequantize_hidden(q_numpy, scale, device, requires_grad)
        # Không quantize: chỉ convert từ fp16
        t = torch.from_numpy(q_numpy.astype(np.float16)).to(device)
        if requires_grad:
            t.requires_grad_(True)
        return t

    def step(self, loss, optimizer):
        """
        Thực hiện backward + optimizer step,
        tự động dùng GradScaler nếu FP16.
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
