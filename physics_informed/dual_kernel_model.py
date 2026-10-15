import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class DualGammaKernel(nn.Module):
    """Fast and slow Gamma responses with window-level rainfall allocation."""

    def __init__(
        self,
        max_lag: int = 30,
        fast_kernel_config: Optional[dict] = None,
        slow_kernel_config: Optional[dict] = None,
    ):
        super().__init__()
        self.max_lag = max_lag
        if fast_kernel_config is None:
            self.fast_config = {
                "alpha_init": 0.8,
                "beta_init": 3.5,
                "alpha_range": (0.5, 1.2),
                "beta_range": (2.5, 5.0),
            }
        else:
            self.fast_config = fast_kernel_config
        if slow_kernel_config is None:
            self.slow_config = {
                "alpha_init": 3.5,
                "beta_init": 0.4,
                "alpha_range": (2.5, 5.0),
                "beta_range": (0.2, 0.8),
            }
        else:
            self.slow_config = slow_kernel_config
        self.alpha_fast = nn.Parameter(torch.tensor(self.fast_config["alpha_init"]))
        self.beta_fast = nn.Parameter(torch.tensor(self.fast_config["beta_init"]))
        self.alpha_slow = nn.Parameter(torch.tensor(self.slow_config["alpha_init"]))
        self.beta_slow = nn.Parameter(torch.tensor(self.slow_config["beta_init"]))
        self.rain_intensity_weights = nn.Parameter(torch.tensor([0.7, 0.3]))
        self.rain_threshold_low = nn.Parameter(torch.tensor(0.2))
        self.rain_threshold_high = nn.Parameter(torch.tensor(0.6))
        self.moisture_effect = nn.Parameter(torch.tensor(0.3))
        self.base_weights = nn.Parameter(torch.tensor([0.2, 0.8]))

    def get_gamma_weights(self, kernel_type: str) -> torch.Tensor:
        """Return normalized weights for the selected lag kernel."""
        if kernel_type == "fast":
            config = self.fast_config
            alpha = torch.clamp(
                self.alpha_fast,
                min=config["alpha_range"][0],
                max=config["alpha_range"][1],
            )
            beta = torch.clamp(
                self.beta_fast, min=config["beta_range"][0], max=config["beta_range"][1]
            )
        elif kernel_type == "slow":
            config = self.slow_config
            alpha = torch.clamp(
                self.alpha_slow,
                min=config["alpha_range"][0],
                max=config["alpha_range"][1],
            )
            beta = torch.clamp(
                self.beta_slow, min=config["beta_range"][0], max=config["beta_range"][1]
            )
        else:
            raise ValueError("kernel_type must be 'fast' or 'slow'")
        t = torch.arange(self.max_lag, dtype=torch.float32, device=alpha.device)
        gamma_weights = torch.pow(t + 1e-06, alpha - 1) * torch.exp(-beta * t)
        gamma_weights = gamma_weights / (torch.sum(gamma_weights) + 1e-08)
        return gamma_weights

    def compute_rainfall_features(
        self, rainfall_sequence: torch.Tensor
    ) -> torch.Tensor:
        """Return the latest and seven-day mean normalized rainfall."""
        batch_size, seq_len, _ = rainfall_sequence.shape
        current_rainfall = rainfall_sequence[:, -1, 0]
        lookback = min(7, seq_len)
        recent_rainfall = torch.mean(rainfall_sequence[:, -lookback:, 0], dim=1)
        current_rainfall = torch.clamp(current_rainfall, min=0, max=1)
        recent_rainfall = torch.clamp(recent_rainfall, min=0, max=1)
        features = torch.stack([current_rainfall, recent_rainfall], dim=1)
        return features

    def compute_dual_weights(self, rainfall_sequence: torch.Tensor) -> torch.Tensor:
        """Allocate rainfall between two responses using recent rainfall."""
        batch_size, seq_len, _ = rainfall_sequence.shape
        device = rainfall_sequence.device
        current_rain = rainfall_sequence[:, -1, 0]
        lookback = min(7, seq_len)
        antecedent_rain = torch.mean(rainfall_sequence[:, -lookback:, 0], dim=1)
        rain_thresh_low = torch.clamp(self.rain_threshold_low, min=0.05, max=0.4)
        rain_thresh_high = torch.clamp(self.rain_threshold_high, min=0.4, max=0.9)
        moisture_effect = torch.clamp(self.moisture_effect, min=0.0, max=0.8)
        base_weights = F.softmax(self.base_weights, dim=0)
        intensity_weights = F.softmax(self.rain_intensity_weights, dim=0)
        intensity_factor = torch.sigmoid((current_rain - rain_thresh_low) * 10)
        high_intensity_factor = torch.sigmoid((current_rain - rain_thresh_high) * 10)
        moisture_factor = torch.sigmoid((antecedent_rain - 0.3) * 8)
        weights = torch.zeros(batch_size, 2, device=device)
        for i in range(batch_size):
            w_base = base_weights.clone()
            if current_rain[i] > rain_thresh_low:
                w_intensity = intensity_weights.clone()
                blend_factor = intensity_factor[i] * (1 + high_intensity_factor[i])
                w_base = (
                    1 - blend_factor * 0.7
                ) * w_base + blend_factor * 0.7 * w_intensity
            if antecedent_rain[i] > 0.1:
                moisture_adj = moisture_factor[i] * moisture_effect
                w_base[0] += moisture_adj * 0.3
                w_base[1] -= moisture_adj * 0.3
            w_base = torch.clamp(w_base, min=0.05, max=0.9)
            w_base = w_base / torch.sum(w_base)
            weights[i] = w_base
        return weights

    def forward(self, rainfall_sequence: torch.Tensor) -> torch.Tensor:
        """Map rainfall (batch, steps, 1) to a response of the same shape."""
        batch_size, seq_len, _ = rainfall_sequence.shape
        device = rainfall_sequence.device
        weights = self.compute_dual_weights(rainfall_sequence)
        fast_gamma_weights = self.get_gamma_weights("fast")
        slow_gamma_weights = self.get_gamma_weights("slow")
        kernel_outputs = []
        fast_weight = weights[:, 0:1].unsqueeze(-1)
        fast_allocated_rainfall = rainfall_sequence * fast_weight
        fast_allocated_rainfall = torch.clamp(fast_allocated_rainfall, min=0, max=1)
        fast_rain = fast_allocated_rainfall.squeeze(-1).unsqueeze(1)
        fast_kernel = fast_gamma_weights.flip(0).view(1, 1, -1)
        fast_output = F.conv1d(fast_rain, fast_kernel, padding=self.max_lag - 1)
        fast_output = fast_output[:, :, :seq_len]
        kernel_outputs.append(fast_output)
        slow_weight = weights[:, 1:2].unsqueeze(-1)
        slow_allocated_rainfall = rainfall_sequence * slow_weight
        slow_allocated_rainfall = torch.clamp(slow_allocated_rainfall, min=0, max=1)
        slow_rain = slow_allocated_rainfall.squeeze(-1).unsqueeze(1)
        slow_kernel = slow_gamma_weights.flip(0).view(1, 1, -1)
        slow_output = F.conv1d(slow_rain, slow_kernel, padding=self.max_lag - 1)
        slow_output = slow_output[:, :, :seq_len]
        kernel_outputs.append(slow_output)
        fused_baseline = torch.zeros_like(kernel_outputs[0])
        for k in range(2):
            fused_baseline += kernel_outputs[k]
        fused_baseline = fused_baseline.transpose(1, 2)
        return fused_baseline

    def get_kernel_params(self) -> dict:
        """Return clamped kernel parameters used in the forward pass."""
        with torch.no_grad():
            fast_alpha = torch.clamp(
                self.alpha_fast,
                min=self.fast_config["alpha_range"][0],
                max=self.fast_config["alpha_range"][1],
            )
            fast_beta = torch.clamp(
                self.beta_fast,
                min=self.fast_config["beta_range"][0],
                max=self.fast_config["beta_range"][1],
            )
            slow_alpha = torch.clamp(
                self.alpha_slow,
                min=self.slow_config["alpha_range"][0],
                max=self.slow_config["alpha_range"][1],
            )
            slow_beta = torch.clamp(
                self.beta_slow,
                min=self.slow_config["beta_range"][0],
                max=self.slow_config["beta_range"][1],
            )
            return {
                "fast_kernel": (float(fast_alpha.item()), float(fast_beta.item())),
                "slow_kernel": (float(slow_alpha.item()), float(slow_beta.item())),
                "rain_threshold_low": float(
                    torch.clamp(self.rain_threshold_low, min=0.05, max=0.4).item()
                ),
                "rain_threshold_high": float(
                    torch.clamp(self.rain_threshold_high, min=0.4, max=0.9).item()
                ),
                "moisture_effect": float(
                    torch.clamp(self.moisture_effect, min=0.0, max=0.8).item()
                ),
                "base_weights": F.softmax(self.base_weights, dim=0)
                .cpu()
                .numpy()
                .tolist(),
                "intensity_weights": F.softmax(self.rain_intensity_weights, dim=0)
                .cpu()
                .numpy()
                .tolist(),
            }

    def get_allocation_weights(self, rainfall_sequence: torch.Tensor) -> torch.Tensor:
        """Return one allocation vector per input window."""
        return self.compute_dual_weights(rainfall_sequence)


class DualKernelPhysicsModel(nn.Module):
    """Two Gamma kernels, an LSTM, and allocation features."""

    def __init__(
        self,
        rainfall_seq_len: int = 30,
        flow_seq_len: int = 30,
        pred_len: int = 1,
        max_lag: int = 30,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.rainfall_seq_len = rainfall_seq_len
        self.flow_seq_len = flow_seq_len
        self.pred_len = pred_len
        self.dual_gamma_kernel = DualGammaKernel(max_lag=max_lag)
        self.flow_encoder = nn.LSTM(1, hidden_dim, num_layers=1, batch_first=True)
        self.rainfall_proj = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        fusion_input_dim = hidden_dim * 2 + 1 + 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, pred_len),
            nn.Softplus(),
        )

    def forward(
        self,
        rainfall_sequence: torch.Tensor,
        flow_sequence: torch.Tensor,
        flow_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict flow from rain and flow tensors shaped (batch, steps, 1)."""
        batch_size = rainfall_sequence.size(0)
        baseline_flow = self.dual_gamma_kernel(rainfall_sequence)
        flow_features, _ = self.flow_encoder(flow_sequence)
        flow_last = flow_features[:, -1, :]
        rainfall_features = self.rainfall_proj(rainfall_sequence)
        rainfall_mean = torch.mean(rainfall_features, dim=1)
        dual_weights = self.dual_gamma_kernel.get_allocation_weights(rainfall_sequence)
        baseline_last = baseline_flow[:, -1, 0].unsqueeze(-1)
        fused = torch.cat(
            [flow_last, rainfall_mean, baseline_last, dual_weights], dim=-1
        )
        fused = self.fusion(fused)
        prediction = self.predictor(fused)
        return prediction

    def get_kernel_info(self) -> dict:
        """Return kernel parameters and response labels for inspection."""
        params = self.dual_gamma_kernel.get_kernel_params()
        info = {
            "model_type": "Dual Kernel",
            "num_kernels": 2,
            "kernel_params": params,
            "kernel_types": ["Fast (Surface Runoff)", "Slow (Groundwater)"],
        }
        return info

    def get_allocation_weights(self, rainfall_sequence: torch.Tensor) -> torch.Tensor:
        """Return one allocation vector per input window."""
        return self.dual_gamma_kernel.get_allocation_weights(rainfall_sequence)


def initialize_dual_kernel_model(model: nn.Module):
    """Apply the initialization used by the original research implementation."""
    for name, param in model.named_parameters():
        if "weight" in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.normal_(param, 0, 0.01)
        elif "bias" in name:
            nn.init.constant_(param, 0)
        elif any(
            (
                x in name
                for x in [
                    "alpha_",
                    "beta_",
                    "rain_threshold",
                    "moisture_effect",
                    "base_weights",
                    "rain_intensity_weights",
                ]
            )
        ):
            continue


def count_dual_kernel_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum((p.numel() for p in model.parameters() if p.requires_grad))
