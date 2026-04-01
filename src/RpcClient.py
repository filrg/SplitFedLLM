import time
import pickle
import re
import os
import torch
import src.Log
from src.fine_tune.GPT2 import Ft_GPT2
from src.fine_tune.Llama import Ft_Llama
from src.fine_tune.Bert import Ft_Bert
from src.dataset.dataloader import dataloader
from src.model.GPT2 import GPT2
from src.model.Llama import Llama
from src.model.Bert import Bert
from src.Optimizer import (
    OptimizationBundle,
    build_qlora_config, apply_qlora,
    cast_model_precision,
    enable_gradient_checkpointing,
)
from peft import LoraConfig, TaskType, get_peft_model


def _build_lora_config(fine_tune_config: dict, model_name: str):
    r     = fine_tune_config["LoRA"]["r"]
    alpha = fine_tune_config["LoRA"]["alpha"]
    if model_name == "GPT2":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
            target_modules=["c_attn", "c_proj", "c_fc"],
            fan_in_fan_out=True,
        )
    elif model_name == "Llama":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
    elif model_name == "Bert":
        return LoraConfig(
            task_type="SEQ_CLS",
            r=r, lora_alpha=alpha, lora_dropout=0.1, bias="none",
            target_modules=["query", "key", "value", "dense"],
        )
    return None


def _merge_lora_into_base(base_sd: dict, lora_sd: dict,
                           lora_r: int, lora_alpha: int,
                           model_name: str = "GPT2") -> dict:
    """
    - GPT2  (Conv1D): W=(in,out), A=(r,in), B=(out,r) → delta=(B@A).T * scale
    - Llama/Bert (Linear): W=(out,in), A=(r,in), B=(out,r) → delta=(B@A) * scale
    """
    merged = dict(base_sd)
    scale  = lora_alpha / lora_r
    use_transpose = (model_name == "GPT2")  # BUG 3 FIX

    lora_A, lora_B = {}, {}
    for k, v in lora_sd.items():
        m = re.match(r'base_model\.model\.(.+)\.lora_([AB])\.default\.weight', k)
        if m:
            base_key = m.group(1) + ".weight"
            if m.group(2) == "A":
                lora_A[base_key] = v.float()
            else:
                lora_B[base_key] = v.float()

    applied = 0
    for base_key in lora_A:
        if base_key in lora_B and base_key in merged:
            A = lora_A[base_key]          # (r, in)
            B = lora_B[base_key]          # (out, r)
            W = merged[base_key].float()
            delta = (B @ A).T * scale if use_transpose else (B @ A) * scale
            merged[base_key] = (W + delta).to(merged[base_key].dtype)
            applied += 1

    if "lm_head.weight" not in merged and "wte.weight" in merged:
        merged["lm_head.weight"] = merged["wte.weight"].clone()

    src.Log.print_with_color(f"[LoRA] Applied {applied} deltas vào base weights.", "green")
    return merged


class RpcClient:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id      = client_id
        self.layer_id       = layer_id
        self.channel        = channel
        self.device         = device
        result              = self.channel.queue_declare(queue="", exclusive=True)
        self.callback_queue = result.method.queue
        self.model_train    = None
        self.train_loader   = None
        self.response       = None
        self._refresh_loader = False  

    def wait_response(self):
        reply_queue_name = f"reply_{self.client_id}"
        self.channel.queue_declare(queue=reply_queue_name, durable=False)
        while True:
            method_frame, _, body = self.channel.basic_get(
                queue=reply_queue_name, auto_ack=True
            )
            if body:
                status = self.response_message(body)
                if not status:
                    break
            time.sleep(0.1)

    def response_message(self, body):
        self.response = pickle.loads(body)
        src.Log.print_with_color(
            f"[<<<] Client received: {self.response['message']}", "blue"
        )
        action = self.response["action"]
        if action == "START":
            return self._handle_start()
        elif action == "STOP":
            return False
        return True

    def _handle_start(self):
        resp             = self.response
        model_name       = resp["model_name"]
        cut_layers       = resp["cut_layers"]
        total_block      = resp["total_block"]
        clip_grad_norm   = resp["clip_grad_norm"]
        data_name        = resp["data_name"]
        num_sample       = resp["num_sample"]
        fine_tune_config = resp["fine_tune_config"]
        opt_cfg          = resp.get("opt_config", {})
        batch_size       = resp["batch_size"]
        lr               = resp["lr"]
        weight_decay     = resp["weight_decay"]
        control_count    = resp["control_count"]
        refresh_each_round = resp.get("refresh_each_round", False)

        opt       = OptimizationBundle(opt_cfg)
        use_flash = getattr(opt, "flash_attention", False)
        if use_flash and self.device == "cpu":
            use_flash = False

        klass_map = {"GPT2": GPT2, "Llama": Llama, "Bert": Bert}
        klass     = klass_map.get(model_name, GPT2)

        if self.layer_id == 1:
            model = klass(layer_id=1, n_block=cut_layers, use_flash=use_flash)
        else:
            model = klass(layer_id=2, n_block=total_block - cut_layers, use_flash=use_flash)

        # ── 2. Load weights từ local file riêng của từng layer ───────────────
  
        layer_file      = f"{model_name}_layer{self.layer_id}.pt"
        pretrained_file = f"{model_name}.pt"
        base_file       = f"{model_name}_base.pt"
        load_file       = layer_file if os.path.exists(layer_file) else pretrained_file

        # ACCUMULATION FIX: Snapshot base weights 1 lần duy nhất.
        # Mọi round đều merge LoRA vào base gốc, không cộng dồn delta.
        if not os.path.exists(base_file) and os.path.exists(pretrained_file):
            import shutil
            shutil.copy2(pretrained_file, base_file)
            src.Log.print_with_color(f"[INFO] Created base snapshot: {base_file}", "green")

        if os.path.exists(load_file):
            src.Log.print_with_color(
                f"[INFO] Layer {self.layer_id}: Loading {load_file}", "green"
            )
            full_sd = torch.load(load_file, map_location="cpu")  
            new_sd  = {}

            if self.layer_id == 1:
                for k, v in full_sd.items():
                    if k.startswith("h."):
                        idx = int(k.split(".")[1])
                        if idx < cut_layers:
                            new_sd[k] = v
                    elif k.startswith(("wte", "wpe")):
                        new_sd[k] = v
            else:
                for k, v in full_sd.items():
                    if k.startswith("h."):
                        idx = int(k.split(".")[1])
                        if idx >= cut_layers:
                            new_k = k.replace(f"h.{idx}", f"h.{idx - cut_layers}")
                            new_sd[new_k] = v
                    elif k.startswith(("ln_f", "lm_head")):
                        new_sd[k] = v

            missing, unexpected = model.load_state_dict(new_sd, strict=False)
            src.Log.print_with_color(
                f"[DEBUG] Layer {self.layer_id} missing={len(missing)} "
                f"unexpected={len(unexpected)}", "blue"
            )
        else:
            src.Log.print_with_color(
                f"[WARN] {load_file} not found — using random init", "yellow"
            )

        # LoRA adapter
        ft_name     = fine_tune_config.get("name", "LoRA")
        peft_config = None

        if fine_tune_config.get("enable", False):
            if ft_name == "QLoRA":
                _, peft_config = build_qlora_config(
                    fine_tune_config["QLoRA"], model, model_name
                )
                if peft_config:
                    model = apply_qlora(model, peft_config, model_name)
                else:
                    peft_config = _build_lora_config(fine_tune_config, model_name)
                    if peft_config:
                        model = get_peft_model(model, peft_config)
            else:
                peft_config = _build_lora_config(fine_tune_config, model_name)
                if peft_config:
                    model = get_peft_model(model, peft_config)
                    if model_name == "Bert" and self.layer_id == 2:
                        for p in model.classifier.parameters():
                            p.requires_grad = True
                    model.print_trainable_parameters()

        if ft_name != "QLoRA":
            model = cast_model_precision(model, opt.precision)
        if opt.gradient_checkpointing:
            model = enable_gradient_checkpointing(model)

        model.to(self.device)

        ft_map = {"GPT2": Ft_GPT2, "Llama": Ft_Llama, "Bert": Ft_Bert}
        self.model_train = ft_map.get(model_name, Ft_GPT2)(
            self.client_id, self.layer_id, self.channel, self.device
        )

        if refresh_each_round:
            self.train_loader = None

        if self.layer_id == 1 and self.train_loader is None:
            self.train_loader = dataloader(
                model_name, data_name, batch_size, num_sample, train=True
            )

        if self.layer_id == 1:
            result, size = self.model_train.first_layer(
                model, lr, weight_decay, clip_grad_norm,
                control_count, self.train_loader, opt=opt
            )
        else:
            result, size = self.model_train.last_layer(
                model, lr, weight_decay, clip_grad_norm, opt=opt
            )

        if fine_tune_config.get("enable", False) and peft_config is not None:
            if not result:
                src.Log.print_with_color(
                    f"[WARN] Layer {self.layer_id}: Training failed, bỏ qua lưu LoRA.",
                    "yellow"
                )
            else:
                lora_sd = {
                    k: v.detach().cpu().float()
                    for k, v in model.state_dict().items()
                    if "lora" in k.lower()
                }
                src.Log.print_with_color(
                    f"[INFO] Layer {self.layer_id}: {len(lora_sd)} LoRA keys collected.",
                    "green"
                )


                # ACCUMULATION FIX: Luôn merge vào base gốc (không phải layer file đã có delta)
                merge_src = base_file if os.path.exists(base_file) else pretrained_file
                if os.path.exists(merge_src):
                    base_sd = torch.load(merge_src, map_location="cpu")
                    if "wte.weight" in base_sd:
                        lora_r     = fine_tune_config.get("LoRA", {}).get("r", 8)
                        lora_alpha = fine_tune_config.get("LoRA", {}).get("alpha", 16)
                        merged_sd  = _merge_lora_into_base(
                            base_sd, lora_sd, lora_r, lora_alpha, model_name
                        )

  
                        partial_sd = {}
                        if self.layer_id == 1:
                            for k, v in merged_sd.items():
                                if k.startswith(("wte", "wpe")):
                                    partial_sd[k] = v
                                elif k.startswith("h."):
                                    idx = int(k.split(".")[1])
                                    if idx < cut_layers:
                                        partial_sd[k] = v
                        else:  
                            for k, v in merged_sd.items():
                                if k.startswith("h."):
                                    idx = int(k.split(".")[1])
                                    if idx >= cut_layers:
                                        new_k = k.replace(f"h.{idx}", f"h.{idx - cut_layers}")
                                        partial_sd[new_k] = v
                                elif k.startswith(("ln_f", "lm_head")):
                                    partial_sd[k] = v

                        torch.save(partial_sd, layer_file)
                        size_mb = os.path.getsize(layer_file) / 1e6
                        src.Log.print_with_color(
                            f"[INFO] Layer {self.layer_id}: Saved {len(partial_sd)} keys "
                            f"→ {layer_file} ({size_mb:.1f} MB)", "green"
                        )
                    else:
                        src.Log.print_with_color(
                            f"[WARN] Layer {self.layer_id}: {merge_src} không có base weights.",
                            "yellow"
                        )
                else:
                    src.Log.print_with_color(
                        f"[WARN] Layer {self.layer_id}: Không tìm thấy {merge_src} để merge.",
                        "yellow"
                    )


        ready_msg = {
            "action":    "READY",
            "client_id": self.client_id,
            "layer_id":  self.layer_id,
            "message":   f"Layer {self.layer_id} saved LoRA, ready for next round.",
        }
        src.Log.print_with_color(
            f"[>>>] Layer {self.layer_id}: Gửi READY về server.", "red"
        )
        self.send_to_server(ready_msg)
        return True

    def send_to_server(self, message):
        self.response = None
        self.channel.queue_declare("rpc_queue", durable=False)
        self.channel.basic_publish(
            exchange="", routing_key="rpc_queue", body=pickle.dumps(message)
        )
        return self.response