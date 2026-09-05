from dataclasses import dataclass

import torch.nn.functional as F


@dataclass(frozen=True)
class TrainingAdapter:
    uses_attention_mask: bool = False
    label_key: str = "labels"
    token_loss: bool = False
    shift_labels: bool = False
    ignore_index: int = -100

    def forward(self, model, inputs, attention_mask=None):
        if self.uses_attention_mask:
            return model(input_ids=inputs, attention_mask=attention_mask)
        return model(input_ids=inputs), None

    def loss(self, output, labels):
        if self.shift_labels:
            output = output[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
        if self.token_loss:
            output = output.reshape(-1, output.size(-1))
            labels = labels.reshape(-1)
        return F.cross_entropy(output, labels, ignore_index=self.ignore_index)


# Preserve the existing testbed objectives, including GPT-2's EOS padding
# and Llama's unshifted input-token targets. Changing objectives is separate
# from changing the distributed scheduler.
_ADAPTERS = {
    "Bert": TrainingAdapter(),
    "GPT2": TrainingAdapter(uses_attention_mask=True, token_loss=True,
                            shift_labels=True, ignore_index=50256),
    "Llama": TrainingAdapter(uses_attention_mask=True, label_key="input_ids",
                             token_loss=True),
}


def register_adapter(model_name, adapter):
    """Register an adapter implementing forward, loss, label_key and mask usage."""
    _ADAPTERS[model_name] = adapter


def get_adapter(model_name):
    try:
        return _ADAPTERS[model_name]
    except KeyError as exc:
        raise ValueError(f"No training adapter registered for {model_name!r}") from exc
