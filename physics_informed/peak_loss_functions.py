import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class PeakWeightedLoss(nn.Module):
    """Squared error with extra weight on upper-quantile targets."""

    def __init__(
        self, peak_threshold_percentile: float = 0.8, peak_weight: float = 5.0
    ):
        super().__init__()
        self.peak_threshold_percentile = peak_threshold_percentile
        self.peak_weight = peak_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for predictions and targets with matching shapes."""
        mse_loss = torch.mean((predictions - targets) ** 2)
        target_values = targets.flatten()
        peak_threshold = torch.quantile(target_values, self.peak_threshold_percentile)
        is_peak = targets >= peak_threshold
        if torch.any(is_peak):
            peak_predictions = predictions[is_peak]
            peak_targets = targets[is_peak]
            peak_loss = torch.mean((peak_predictions - peak_targets) ** 2)
            total_loss = mse_loss + self.peak_weight * peak_loss
        else:
            total_loss = mse_loss
        return total_loss


class AdaptivePeakLoss(nn.Module):
    """Squared error weighted by peak and extreme-flow quantiles."""

    def __init__(
        self,
        base_weight: float = 1.0,
        peak_weight: float = 10.0,
        extreme_weight: float = 20.0,
        peak_percentile: float = 0.8,
        extreme_percentile: float = 0.95,
    ):
        super().__init__()
        self.base_weight = base_weight
        self.peak_weight = peak_weight
        self.extreme_weight = extreme_weight
        self.peak_percentile = peak_percentile
        self.extreme_percentile = extreme_percentile

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for predictions and targets with matching shapes."""
        target_values = targets.flatten()
        peak_threshold = torch.quantile(target_values, self.peak_percentile)
        extreme_threshold = torch.quantile(target_values, self.extreme_percentile)
        errors = (predictions - targets) ** 2
        weights = torch.ones_like(targets) * self.base_weight
        peak_mask = targets >= peak_threshold
        weights[peak_mask] = self.peak_weight
        extreme_mask = targets >= extreme_threshold
        weights[extreme_mask] = self.extreme_weight
        weighted_loss = torch.mean(errors * weights)
        return weighted_loss


class SampleSizeAwarePeakLoss(nn.Module):
    """Peak weighting adjusted to the number of training samples."""

    def __init__(
        self,
        sample_size: int,
        max_samples: int = 10000,
        base_weight: float = 1.0,
        peak_weight_base: float = 5.0,
        peak_weight_max: float = 20.0,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.max_samples = max_samples
        self.base_weight = base_weight
        sample_ratio = min(sample_size / max_samples, 1.0)
        self.peak_weight = peak_weight_base + (peak_weight_max - peak_weight_base) * (
            1 - sample_ratio
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for predictions and targets with matching shapes."""
        target_values = targets.flatten()
        if self.sample_size < 1000:
            peak_percentile = 0.7
        elif self.sample_size < 5000:
            peak_percentile = 0.8
        else:
            peak_percentile = 0.85
        peak_threshold = torch.quantile(target_values, peak_percentile)
        mse_loss = torch.mean((predictions - targets) ** 2)
        peak_mask = targets >= peak_threshold
        if torch.any(peak_mask):
            peak_predictions = predictions[peak_mask]
            peak_targets = targets[peak_mask]
            peak_loss = torch.mean((peak_predictions - peak_targets) ** 2)
            total_loss = mse_loss + self.peak_weight * peak_loss
        else:
            total_loss = mse_loss
        return total_loss


class ComprehensivePeakLoss(nn.Module):
    """Sample-size-dependent weighting for high, middle, and low flows."""

    def __init__(
        self,
        sample_size: int,
        max_samples: int = 10000,
        base_weight: float = 1.0,
        peak_weight_base: float = 3.0,
        peak_weight_max: float = 5.0,
        valley_weight_base: float = 4.0,
        valley_weight_max: float = 8.0,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.max_samples = max_samples
        self.base_weight = base_weight
        sample_ratio = min(sample_size / max_samples, 1.0)
        self.peak_weight = peak_weight_base + (peak_weight_max - peak_weight_base) * (
            1 - sample_ratio
        )
        self.valley_weight = valley_weight_base + (
            valley_weight_max - valley_weight_base
        ) * (1 - sample_ratio)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for predictions and targets with matching shapes."""
        target_values = targets.flatten()
        if self.sample_size < 1000:
            peak_percentile = 0.75
            valley_percentile = 0.5
        elif self.sample_size < 5000:
            peak_percentile = 0.8
            valley_percentile = 0.4
        else:
            peak_percentile = 0.85
            valley_percentile = 0.3
        peak_threshold = torch.quantile(target_values, peak_percentile)
        valley_threshold = torch.quantile(target_values, valley_percentile)
        errors = (predictions - targets) ** 2
        weights = torch.ones_like(targets) * self.base_weight
        peak_mask = targets >= peak_threshold
        weights[peak_mask] = self.peak_weight
        valley_mask = targets <= valley_threshold
        weights[valley_mask] = self.valley_weight
        medium_mask = (targets > valley_threshold) & (targets < peak_threshold)
        weights[medium_mask] = 3.0
        weighted_loss = torch.mean(errors * weights)
        return weighted_loss


def get_peak_loss_function(loss_type: str = "comprehensive", **kwargs) -> nn.Module:
    """Build one of the retained peak-weighted loss functions."""
    if loss_type == "peak_weighted":
        return PeakWeightedLoss(**kwargs)
    elif loss_type == "adaptive":
        return AdaptivePeakLoss(**kwargs)
    elif loss_type == "sample_aware":
        return SampleSizeAwarePeakLoss(**kwargs)
    elif loss_type == "comprehensive":
        return ComprehensivePeakLoss(**kwargs)
    else:
        return nn.MSELoss()
