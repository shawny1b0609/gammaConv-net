"""Short names for the three research architectures."""

from physics_informed.simplified_physics_model import SimplifiedPhysicsModel
from physics_informed.dual_kernel_model import DualKernelPhysicsModel
from physics_informed.improved_water_balanced_kernel import ImprovedWaterBalancedModel

MODELS = {
    "single": SimplifiedPhysicsModel,
    "dual": DualKernelPhysicsModel,
    "triple": ImprovedWaterBalancedModel,
}


def make_model(name="dual", hidden=64, seq=30, lag=30):
    """Build a one-day-ahead model; inputs have shape (batch, seq, 1)."""
    if name not in MODELS:
        raise ValueError(f"Unknown model: {name}. Choose single, dual, or triple.")
    if hidden < 4 or seq < 1 or lag < 1:
        raise ValueError("hidden must be >= 4; seq and lag must be positive.")
    return MODELS[name](
        rainfall_seq_len=seq,
        flow_seq_len=seq,
        pred_len=1,
        max_lag=lag,
        hidden_dim=hidden,
    )
