import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SimplifiedGammaKernel(nn.Module):
    """A normalized, finite-lag Gamma response kernel."""

    def __init__(
        self, alpha_init: float = 2.0, beta_init: float = 1.0, max_lag: int = 30
    ):
        super().__init__()
        self.max_lag = max_lag
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def get_gamma_weights(self) -> torch.Tensor:
        """Return normalized weights for the selected lag kernel."""
        alpha = torch.clamp(self.alpha, min=0.5, max=5.0)
        beta = torch.clamp(self.beta, min=0.1, max=3.0)
        t = torch.arange(self.max_lag, dtype=torch.float32, device=self.alpha.device)
        gamma_weights = torch.pow(t + 1e-06, alpha - 1) * torch.exp(-beta * t)
        gamma_weights = gamma_weights / (torch.sum(gamma_weights) + 1e-08)
        return gamma_weights

    def forward(self, rainfall_sequence: torch.Tensor) -> torch.Tensor:
        """Map rainfall (batch, steps, 1) to a response of the same shape."""
        batch_size, seq_len, _ = rainfall_sequence.shape
        gamma_weights = self.get_gamma_weights()
        rainfall = rainfall_sequence.squeeze(-1).unsqueeze(1)
        kernel = gamma_weights.flip(0).view(1, 1, -1)
        baseline_flow = F.conv1d(rainfall, kernel, padding=self.max_lag - 1)
        baseline_flow = baseline_flow[:, :, :seq_len].transpose(1, 2)
        return baseline_flow


class SimplifiedPhysicsModel(nn.Module):
    """Single-kernel rainfall response combined with an LSTM flow encoder."""

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
        self.gamma_kernel = SimplifiedGammaKernel(max_lag=max_lag)
        self.flow_encoder = nn.LSTM(1, hidden_dim, num_layers=1, batch_first=True)
        self.rainfall_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
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
        baseline_flow = self.gamma_kernel(rainfall_sequence)
        flow_features, _ = self.flow_encoder(flow_sequence)
        flow_last = flow_features[:, -1, :]
        rainfall_features = self.rainfall_proj(rainfall_sequence)
        rainfall_mean = torch.mean(rainfall_features, dim=1)
        baseline_last = baseline_flow[:, -1, 0].unsqueeze(-1)
        fused = torch.cat([flow_last, rainfall_mean, baseline_last], dim=-1)
        fused = self.fusion(fused)
        prediction = self.predictor(fused)
        return prediction

    def get_gamma_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the raw trainable shape and rate parameters."""
        return (self.gamma_kernel.alpha, self.gamma_kernel.beta)


def initialize_simplified_model(model: nn.Module):
    """Apply the initialization used by the original research implementation."""
    for name, param in model.named_parameters():
        if "weight" in name:
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.normal_(param, 0, 0.01)
        elif "bias" in name:
            nn.init.constant_(param, 0)
        elif "alpha" in name:
            nn.init.constant_(param, 2.0)
        elif "beta" in name:
            nn.init.constant_(param, 1.0)


def count_simplified_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum((p.numel() for p in model.parameters() if p.requires_grad))
