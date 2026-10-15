"""Train a one-day-ahead GammaConv-Net model and save a reusable checkpoint."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import prepare, read_data
from .loss import FlowLoss
from .metrics import scores
from .models import MODELS, make_model
from physics_informed.peak_loss_functions import ComprehensivePeakLoss


@dataclass
class Config:
    data: str = "data/series_a.csv"
    out: str = "runs/dual"
    model: str = "dual"
    epochs: int = 100
    hidden: int = 64
    seq: int = 30
    lag: int = 30
    split: float = 0.818
    fraction: float = 1.0
    batch: int = 32
    lr: float = 5e-4
    wd: float = 1e-4
    patience: int = 50
    seed: int = 42
    device: str = "cpu"
    threads: int = 2
    loss: str = "comprehensive"
    peak: float = 25.0
    low: float = 18.0
    mid: float = 12.0


def predict(model, rain, flow, device="cpu", batch=256):
    """Evaluate windows in batches, without dropout."""
    model.eval()
    values = []
    with torch.no_grad():
        for i in range(0, len(rain), batch):
            values.append(
                model(
                    rain[i : i + batch].to(device), flow[i : i + batch].to(device)
                ).cpu()
            )
    return torch.cat(values).numpy().reshape(-1)


def clone_state(model):
    """Copy parameters: CPU checkpoints must not share live tensor storage."""
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def kernel_info(model):
    if hasattr(model, "get_kernel_info"):
        return model.get_kernel_info()
    kernel = model.gamma_kernel
    return {
        "num_kernels": 1,
        "kernel_params": [
            [
                float(kernel.alpha.detach().clamp(0.5, 5.0).cpu()),
                float(kernel.beta.detach().clamp(0.1, 3.0).cpu()),
            ]
        ],
    }


def run(config):
    if min(config.epochs, config.batch, config.patience, config.threads) < 1:
        raise ValueError("epochs, batch, patience, and threads must be positive.")
    if not np.isfinite(
        [config.lr, config.wd, config.peak, config.low, config.mid]
    ).all():
        raise ValueError("Optimizer and loss settings must be finite.")
    if config.lr <= 0 or config.wd < 0 or min(config.peak, config.low, config.mid) < 0:
        raise ValueError(
            "lr must be positive; weight decay and loss weights nonnegative."
        )
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable. Use --device cpu.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    data = read_data(config.data)
    train, valid, scale, info = prepare(data, config.seq, config.split, config.fraction)
    model = make_model(config.model, config.hidden, config.seq, config.lag).to(device)
    if config.loss == "comprehensive":
        loss_fn = ComprehensivePeakLoss(len(train[0]))
    elif config.loss == "bands":
        loss_fn = FlowLoss(config.peak, config.low, config.mid)
    elif config.loss == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError("loss must be comprehensive, bands, or mse.")
    loader = DataLoader(
        TensorDataset(*train),
        batch_size=config.batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.wd
    )
    # The comprehensive-loss entry point originally used cosine decay.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=min(1e-6, config.lr)
        )
        if config.loss == "comprehensive"
        else None
    )
    obs = data.flow.to_numpy(dtype=float)[info["split_row"] + config.seq :]
    best, best_epoch, stale = np.inf, 0, 0
    best_state, history = None, []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for rain, flow, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(rain.to(device), flow.to(device)), target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}.")
            loss.backward()
            if config.loss == "comprehensive":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(target)
        norm = predict(model, valid[0], valid[1], device)
        pred = (
            norm.astype(float) * (scale["flow"][1] - scale["flow"][0] + 1e-12)
            + scale["flow"][0]
        )
        result = scores(obs, pred)
        history.append({"epoch": epoch, "loss": total / len(train[0]), **result})
        # Minimizing squared error is equivalent to maximizing NSE on fixed targets.
        if result["rmse"] < best:
            best, best_epoch, stale = result["rmse"], epoch, 0
            best_state = clone_state(model)
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
            print(
                f"{config.model}: epoch {epoch}, validation RMSE {result['rmse']:.3f}",
                flush=True,
            )
        if stale >= config.patience:
            break

    model.load_state_dict(best_state)
    model.to("cpu").eval()
    pred = predict(model, valid[0], valid[1]).astype(float)
    pred = pred * (scale["flow"][1] - scale["flow"][0] + 1e-12) + scale["flow"][0]
    result = scores(obs, pred)
    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    params = {
        "name": config.model,
        "hidden": config.hidden,
        "seq": config.seq,
        "lag": config.lag,
    }
    torch.save(
        {
            "model": params,
            "state": best_state,
            "scale": scale,
            "best_epoch": best_epoch,
        },
        out / "model.pt",
    )
    pd.DataFrame(
        {
            "day": data.day.iloc[info["split_row"] + config.seq :].to_numpy(),
            "observed": obs,
            "predicted": pred,
        }
    ).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    settings = asdict(config)
    settings.pop("out")
    settings["data"] = Path(config.data).name
    record = {
        "config": settings,
        "data_sha256": hashlib.sha256(Path(config.data).read_bytes()).hexdigest(),
        "split": info,
        "scale": scale,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "validation": result,
        "parameters": sum(p.numel() for p in model.parameters()),
        "kernels": kernel_info(model),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": device,
            "platform": platform.system(),
        },
    }
    (out / "run.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, **result}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for name, default in asdict(defaults).items():
        choices = {
            "model": list(MODELS),
            "device": ["cpu", "cuda", "auto"],
            "loss": ["comprehensive", "bands", "mse"],
        }.get(name)
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=type(default),
            default=default,
            choices=choices,
        )
    run(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
