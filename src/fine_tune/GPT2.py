"""Compatibility entry point; training is implemented by SplitScheduler."""

from src.fine_tune.adapters import get_adapter
from src.fine_tune.scheduler import SplitScheduler


class Ft_GPT2(SplitScheduler):
    def __init__(self, client_id, layer_id, channel, device):
        super().__init__(client_id, layer_id, channel, device, get_adapter("GPT2"))
