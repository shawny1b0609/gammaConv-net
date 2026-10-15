"""Flow-band loss from the original small-sample experiment."""

import torch
from torch import nn


class FlowLoss(nn.Module):
    def __init__(self, peak=25.0, low=18.0, mid=12.0):
        super().__init__()
        self.weights = (peak, low, mid)

    def forward(self, pred, target):
        error = (pred - target) ** 2
        peak = target > torch.quantile(target, 0.8)
        low = target < torch.quantile(target, 0.3)
        masks = (peak, low, ~(peak | low))
        return sum(
            w * error[m].mean() if m.any() else error.sum() * 0
            for w, m in zip(self.weights, masks)
        )
