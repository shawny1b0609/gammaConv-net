import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class GammaKernel(nn.Module):
    """Gamma kernel used by the earlier correction-based model."""

    def __init__(
        self, alpha_init: float = 2.0, beta_init: float = 1.0, max_lag: int = 30
    ):
        super().__init__()
        self.max_lag = max_lag
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.register_buffer("time_grid", torch.arange(max_lag, dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        """Apply the module to its inputs."""
        alpha = torch.clamp(self.alpha, min=1e-06, max=10.0)
        beta = torch.clamp(self.beta, min=1e-06, max=10.0)
        gamma_weights = torch.pow(self.time_grid + 1e-06, alpha - 1) * torch.exp(
            -beta * self.time_grid
        )
        gamma_weights = gamma_weights / (torch.sum(gamma_weights) + 1e-08)
        return gamma_weights


class RainfallImpulseModule(nn.Module):
    """Rainfall response module from the earlier model."""

    def __init__(self, max_lag: int = 30, hidden_dim: int = 64):
        super().__init__()
        self.max_lag = max_lag
        self.gamma_kernel = GammaKernel(max_lag=max_lag)
        self.rainfall_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, rainfall_sequence: torch.Tensor) -> torch.Tensor:
        """Map rainfall (batch, steps, 1) to a response of the same shape."""
        batch_size, seq_len, _ = rainfall_sequence.shape
        gamma_weights = self.gamma_kernel()
        rainfall_features = self.rainfall_encoder(rainfall_sequence)
        rainfall_features_t = rainfall_features.transpose(1, 2)
        rainfall_features_conv = self.temporal_conv(rainfall_features_t)
        rainfall_features_conv = rainfall_features_conv.transpose(1, 2)
        baseline_flow = torch.zeros(
            batch_size, seq_len, 1, device=rainfall_sequence.device
        )
        for t in range(seq_len):
            flow_contribution = torch.zeros(batch_size, device=rainfall_sequence.device)
            for lag in range(min(t + 1, self.max_lag)):
                if t - lag >= 0:
                    rainfall_val = rainfall_sequence[:, t - lag, 0]
                    weight = gamma_weights[lag]
                    flow_contribution += rainfall_val * weight
            baseline_flow[:, t, 0] = flow_contribution
        return baseline_flow


class FlowCorrectionModule(nn.Module):
    """Adjust a rainfall response using an observed flow value."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.correction_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        observed_flow: torch.Tensor,
        baseline_flow: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the module to its inputs."""
        residual = observed_flow - baseline_flow
        input_features = torch.cat([observed_flow, baseline_flow, delta_t], dim=-1)
        correction = self.correction_mlp(input_features)
        uncertainty = self.uncertainty_mlp(input_features)
        return (correction, uncertainty)


class HydrologicalPhysicsModel(nn.Module):
    """Earlier correction-based architecture, retained for research reference."""

    def __init__(
        self,
        rainfall_seq_len: int = 30,
        flow_seq_len: int = 30,
        pred_len: int = 1,
        max_lag: int = 30,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.rainfall_seq_len = rainfall_seq_len
        self.flow_seq_len = flow_seq_len
        self.pred_len = pred_len
        self.max_lag = max_lag
        self.rainfall_impulse = RainfallImpulseModule(
            max_lag=max_lag, hidden_dim=hidden_dim
        )
        self.flow_correction = FlowCorrectionModule(input_dim=3, hidden_dim=hidden_dim)
        self.flow_encoder = nn.LSTM(
            1, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pred_len),
        )

    def forward(
        self,
        rainfall_sequence: torch.Tensor,
        flow_sequence: torch.Tensor,
        flow_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict flow from rain and flow tensors shaped (batch, steps, 1)."""
        batch_size = rainfall_sequence.size(0)
        baseline_flow = self.rainfall_impulse(rainfall_sequence)
        flow_features, _ = self.flow_encoder(flow_sequence)
        last_flow_features = flow_features[:, -1, :]
        last_baseline_flow = baseline_flow[:, -1, :]
        flow_features_expanded = last_flow_features.unsqueeze(1).expand(
            -1, self.rainfall_seq_len, -1
        )
        fused_features = torch.cat(
            [
                flow_features_expanded,
                torch.zeros(
                    batch_size,
                    self.rainfall_seq_len,
                    last_flow_features.size(-1),
                    device=rainfall_sequence.device,
                ),
            ],
            dim=-1,
        )
        fused_features = self.feature_fusion(fused_features)
        final_features = fused_features[:, -1, :]
        final_baseline = last_baseline_flow
        prediction_input = torch.cat([final_features, final_baseline], dim=-1)
        prediction = self.predictor(prediction_input)
        return prediction

    def get_gamma_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the raw trainable shape and rate parameters."""
        return (
            self.rainfall_impulse.gamma_kernel.alpha,
            self.rainfall_impulse.gamma_kernel.beta,
        )


def initialize_hydrological_model(model: nn.Module):
    """Apply the initialization used by the original research implementation."""
    for name, param in model.named_parameters():
        if "weight" in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.normal_(param, 0, 0.01)
        elif "bias" in name:
            nn.init.constant_(param, 0)
        elif "alpha" in name or "beta" in name:
            pass


def count_hydrological_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum((p.numel() for p in model.parameters() if p.requires_grad))
