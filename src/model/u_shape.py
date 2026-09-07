"""U-shaped partitions and reversible mapping to the original checkpoint keys."""

from torch import nn


def model_class(name):
    from src.model.Bert import Bert
    from src.model.GPT2 import GPT2
    from src.model.Llama import Llama
    try:
        return {"Bert": Bert, "GPT2": GPT2, "Llama": Llama}[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {name}") from exc


def validate_cuts(front_blocks, total_blocks, tail_blocks):
    if any(type(n) is not int for n in (front_blocks, total_blocks, tail_blocks)):
        raise ValueError("Partition sizes must be integers")
    if min(front_blocks, tail_blocks) < 0 or front_blocks + tail_blocks >= total_blocks:
        raise ValueError("U-shape requires nonnegative local cuts and at least one body block")


def build_partition(name, role, front_blocks, total_blocks, tail_blocks=0, **kwargs):
    validate_cuts(front_blocks, total_blocks, tail_blocks)
    klass = model_class(name)
    if role == "owner":
        return nn.ModuleDict({
            "front": klass(layer_id=1, n_block=front_blocks, **kwargs),
            "tail": klass(layer_id=4, n_block=tail_blocks, **kwargs),
        })
    if role == "worker":
        return klass(layer_id=3, n_block=total_blocks - front_blocks - tail_blocks, **kwargs)
    raise ValueError(f"Unknown U-shape role: {role}")


def split_state_dict(state, front_blocks, total_blocks, tail_blocks=0):
    """No temporary full model allocation, and no duplicated head parameters."""
    validate_cuts(front_blocks, total_blocks, tail_blocks)
    owner, body = {}, {}
    front_prefixes = ("embeddings.", "wte.", "wpe.", "embed_tokens.")
    for key, value in state.items():
        parts = key.split(".")
        if parts[0] in ("h", "layers"):
            index = int(parts[1])
            if not 0 <= index < total_blocks:
                raise ValueError(f"Block outside configured model: {key}")
            if index < front_blocks:
                owner["front." + key] = value
            elif index >= total_blocks - tail_blocks:
                parts[1] = str(index - (total_blocks - tail_blocks))
                owner["tail." + ".".join(parts)] = value
            else:
                parts[1] = str(index - front_blocks)
                body[".".join(parts)] = value
        else:
            prefix = "front." if key.startswith(front_prefixes) else "tail."
            owner[prefix + key] = value
    return owner, body


def join_state_dict(owner, body, front_blocks, total_blocks, tail_blocks=0):
    validate_cuts(front_blocks, total_blocks, tail_blocks)
    full = {}
    for partition, state in (("owner", owner), ("worker", body)):
        for key, value in state.items():
            offset = front_blocks
            if partition == "owner":
                role, key = key.split(".", 1)
                if role not in ("front", "tail"):
                    raise ValueError(f"Invalid owner partition: {role}")
                offset = 0 if role == "front" else total_blocks - tail_blocks
            parts = key.split(".")
            if parts[0] in ("h", "layers"):
                parts[1] = str(int(parts[1]) + offset)
            key = ".".join(parts)
            if key in full:
                raise ValueError(f"Duplicate checkpoint key: {key}")
            full[key] = value
    return full


def apply_lora(model, name, role, config):
    """Use generic PEFT wrapping: these custom partitions are not HF task models."""
    from peft import LoraConfig, get_peft_model
    targets = {"Bert": ["query", "key", "value", "dense"],
               "GPT2": ["c_attn", "c_proj", "c_fc"],
               "Llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}[name]
    def wrap(part):
        if not any(key.split(".")[-1] in targets for key, _ in part.named_modules()):
            return part  # A zero-block partition has no transformer LoRA targets.
        return get_peft_model(part, LoraConfig(
            r=config["r"], lora_alpha=config["alpha"], lora_dropout=0.05,
            target_modules=targets, fan_in_fan_out=name == "GPT2", bias="none"))
    if role == "owner":
        model["front"] = wrap(model["front"])
        model["tail"] = wrap(model["tail"])
        # The private output head remains trainable, even when the tail has LoRA blocks.
        for key, param in model["tail"].named_parameters():
            if any(part in key.split(".") for part in ("lm_head", "classifier")):
                param.requires_grad_(True)
        return model
    return wrap(model)


def merge_lora(model, role):
    if role == "owner":
        for part in ("front", "tail"):
            if hasattr(model[part], "merge_and_unload"):
                model[part] = model[part].merge_and_unload()
    elif hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    return model
