import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class ImprovedWaterBalancedKernel(nn.Module):
    """Three response kernels with nonnegative rainfall allocation."""

    def __init__(
        self,
        num_kernels: int = 3,
        max_lag: int = 30,
        kernel_configs: Optional[List[dict]] = None,
    ):
        super().__init__()
        self.num_kernels = num_kernels
        self.max_lag = max_lag
        if kernel_configs is None:
            self.kernel_configs = [
                {
                    "alpha_init": 0.8,
                    "beta_init": 3.0,
                    "alpha_range": (0.5, 1.5),
                    "beta_range": (2.0, 5.0),
                },
                {
                    "alpha_init": 2.0,
                    "beta_init": 1.0,
                    "alpha_range": (1.5, 3.0),
                    "beta_range": (0.8, 2.0),
                },
                {
                    "alpha_init": 4.0,
                    "beta_init": 0.3,
                    "alpha_range": (3.0, 6.0),
                    "beta_range": (0.1, 0.8),
                },
            ]
        else:
            self.kernel_configs = kernel_configs
        self.alphas = nn.ParameterList()
        self.betas = nn.ParameterList()
        for config in self.kernel_configs:
            self.alphas.append(nn.Parameter(torch.tensor(config["alpha_init"])))
            self.betas.append(nn.Parameter(torch.tensor(config["beta_init"])))
        self.rain_intensity_weights = nn.Parameter(torch.tensor([0.6, 0.3, 0.1]))
        self.rain_threshold_low = nn.Parameter(torch.tensor(0.2))
        self.rain_threshold_high = nn.Parameter(torch.tensor(0.6))
        self.moisture_effect = nn.Parameter(torch.tensor(0.3))
        self.base_weights = nn.Parameter(torch.tensor([0.2, 0.3, 0.5]))

    def get_gamma_weights(self, kernel_idx: int) -> torch.Tensor:
        """Return normalized weights for the selected lag kernel."""
        config = self.kernel_configs[kernel_idx]
        alpha = torch.clamp(
            self.alphas[kernel_idx],
            min=config["alpha_range"][0],
            max=config["alpha_range"][1],
        )
        beta = torch.clamp(
            self.betas[kernel_idx],
            min=config["beta_range"][0],
            max=config["beta_range"][1],
        )
        t = torch.arange(self.max_lag, dtype=torch.float32, device=alpha.device)
        gamma_weights = torch.pow(t + 1e-06, alpha - 1) * torch.exp(-beta * t)
        gamma_weights = gamma_weights / (torch.sum(gamma_weights) + 1e-08)
        return gamma_weights

    def compute_improved_weights(self, rainfall_sequence: torch.Tensor) -> torch.Tensor:
        """Allocate rainfall among three responses using recent rainfall."""
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
        weights = torch.zeros(batch_size, self.num_kernels, device=device)
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
                w_base[2] -= moisture_adj * 0.3
            w_base = torch.clamp(w_base, min=0.05, max=0.9)
            w_base = w_base / torch.sum(w_base)
            weights[i] = w_base
        return weights

    def forward(self, rainfall_sequence: torch.Tensor) -> torch.Tensor:
        """Map rainfall (batch, steps, 1) to a response of the same shape."""
        batch_size, seq_len, _ = rainfall_sequence.shape
        device = rainfall_sequence.device
        weights = self.compute_improved_weights(rainfall_sequence)
        kernel_outputs = []
        for k in range(self.num_kernels):
            gamma_weights = self.get_gamma_weights(k)
            kernel_k_weight = weights[:, k : k + 1].unsqueeze(-1)
            allocated_rainfall = rainfall_sequence * kernel_k_weight
            allocated_rain = allocated_rainfall.squeeze(-1).unsqueeze(1)
            kernel = gamma_weights.flip(0).view(1, 1, -1)
            baseline_k = F.conv1d(allocated_rain, kernel, padding=self.max_lag - 1)
            baseline_k = baseline_k[:, :, :seq_len]
            kernel_outputs.append(baseline_k)
        fused_baseline = torch.zeros_like(kernel_outputs[0])
        for k in range(self.num_kernels):
            fused_baseline += kernel_outputs[k]
        fused_baseline = fused_baseline.transpose(1, 2)
        return fused_baseline

    def get_kernel_params(self) -> List[Tuple[float, float]]:
        """Return clamped kernel parameters used in the forward pass."""
        params = []
        for k in range(self.num_kernels):
            config = self.kernel_configs[k]
            alpha = torch.clamp(
                self.alphas[k],
                min=config["alpha_range"][0],
                max=config["alpha_range"][1],
            )
            beta = torch.clamp(
                self.betas[k], min=config["beta_range"][0], max=config["beta_range"][1]
            )
            params.append((float(alpha.item()), float(beta.item())))
        return params

    def get_allocation_weights(self, rainfall_sequence: torch.Tensor) -> torch.Tensor:
        """Return one allocation vector per input window."""
        return self.compute_improved_weights(rainfall_sequence)

    def get_allocation_info(self) -> dict:
        """Return the fitted rainfall-allocation parameters."""
        with torch.no_grad():
            return {
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


class ImprovedWaterBalancedModel(nn.Module):
    """Three Gamma kernels combined with rainfall and flow features."""

    def __init__(
        self,
        rainfall_seq_len: int = 30,
        flow_seq_len: int = 30,
        pred_len: int = 1,
        max_lag: int = 30,
        hidden_dim: int = 32,
        num_kernels: int = 3,
    ):
        super().__init__()
        self.rainfall_seq_len = rainfall_seq_len
        self.flow_seq_len = flow_seq_len
        self.pred_len = pred_len
        self.num_kernels = num_kernels
        self.improved_water_kernel = ImprovedWaterBalancedKernel(
            num_kernels=num_kernels, max_lag=max_lag
        )
        self.flow_encoder = nn.LSTM(1, hidden_dim, num_layers=1, batch_first=True)
        self.rainfall_proj = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(), nn.Dropout(0.1)
        )
        fusion_input_dim = hidden_dim * 2 + 1
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
        baseline_flow = self.improved_water_kernel(rainfall_sequence)
        flow_features, _ = self.flow_encoder(flow_sequence)
        flow_last = flow_features[:, -1, :]
        rainfall_features = self.rainfall_proj(rainfall_sequence)
        rainfall_mean = torch.mean(rainfall_features, dim=1)
        baseline_last = baseline_flow[:, -1, 0].unsqueeze(-1)
        fused = torch.cat([flow_last, rainfall_mean, baseline_last], dim=-1)
        fused = self.fusion(fused)
        prediction = self.predictor(fused)
        return prediction

    def get_kernel_info(self) -> dict:
        """Return kernel parameters and response labels for inspection."""
        params = self.improved_water_kernel.get_kernel_params()
        allocation_info = self.improved_water_kernel.get_allocation_info()
        info = {
            "num_kernels": self.num_kernels,
            "kernel_params": params,
            "kernel_types": [
                "Fast (Surface)",
                "Medium (Subsurface)",
                "Slow (Groundwater)",
            ],
            "allocation_mechanism": allocation_info,
        }
        return info

    def get_allocation_weights(self, rainfall_sequence: torch.Tensor) -> torch.Tensor:
        """Return one allocation vector per input window."""
        return self.improved_water_kernel.get_allocation_weights(rainfall_sequence)


def initialize_improved_water_balanced_model(model: nn.Module):
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
                    "rain_threshold",
                    "moisture_effect",
                    "base_weights",
                    "rain_intensity_weights",
                ]
            )
        ):
            continue
