"""Plot predictions, fitted Gamma kernels, and rainfall allocation in English."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .predict import load_model


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(run, out):
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model, state = load_model(run / "model.pt")
    result = json.loads((run / "run.json").read_text())
    data = pd.read_csv(run / "predictions.csv")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "axes.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 5.2),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(data.day, data.observed, color="#333333", lw=1.2, label="Observed")
    axes[0].plot(
        data.day, data.predicted, color=colors[0], lw=1.1, ls="--", label="Predicted"
    )
    axes[0].set_ylabel(r"Flow (m$^3$ s$^{-1}$)")
    axes[0].legend(loc="upper right", ncols=2)
    nse = result["validation"]["nse"]
    label = (
        f"NSE = {nse:.3f}" if nse is not None else "NSE undefined (constant targets)"
    )
    axes[0].set_title(
        f"{result['config']['model'].capitalize()} kernel model | {label} | validation example",
        loc="left",
        fontsize=11,
    )
    axes[1].axhline(0, color="#555555", lw=0.7)
    axes[1].plot(data.day, data.predicted - data.observed, color=colors[1], lw=0.8)
    axes[1].set_ylabel("Error\n(pred. - obs.)")
    axes[1].set_xlabel("Anonymized day index")
    save(fig, out, "prediction")

    if hasattr(model, "gamma_kernel"):
        kernel = model.gamma_kernel
        weights, labels = [kernel.get_gamma_weights()], ["Single"]
    elif hasattr(model, "dual_gamma_kernel"):
        kernel = model.dual_gamma_kernel
        weights = [kernel.get_gamma_weights(k) for k in ("fast", "slow")]
        labels = ["Fast", "Slow"]
    else:
        kernel = model.improved_water_kernel
        weights = [kernel.get_gamma_weights(k) for k in range(3)]
        labels = ["Fast", "Intermediate", "Slow"]
    fig, ax = plt.subplots(figsize=(6.5, 3.3), layout="constrained")
    for i, (weight, label) in enumerate(zip(weights, labels)):
        value = weight.detach().numpy()
        ax.plot(
            np.arange(len(value)),
            value,
            color=colors[i],
            marker=["o", "s", "^"][i],
            ms=3,
            lw=1.3,
            label=label,
        )
    ax.set(
        xlabel="Lag (days)",
        ylabel="Normalized kernel weight",
        title="Fitted discrete Gamma responses",
    )
    ax.legend()
    save(fig, out, "kernels")

    if len(weights) == 1:
        return
    # The seven-day mean includes the latest rainfall value. Mask impossible pairs.
    current, mean = np.meshgrid(np.linspace(0, 1, 35), np.linspace(0, 1, 35))
    previous = (7 * mean - current) / 6
    valid = (previous >= 0) & (previous <= 1)
    seq = max(7, state["model"]["seq"])
    rain = torch.zeros(current.size, seq, 1)
    rain[:, -7:-1, 0] = torch.tensor(
        previous.clip(0, 1).reshape(-1, 1), dtype=torch.float32
    )
    rain[:, -1, 0] = torch.tensor(current.ravel(), dtype=torch.float32)
    with torch.no_grad():
        values = kernel.get_allocation_weights(rain).numpy()
    fig, axes = plt.subplots(
        1,
        len(weights),
        figsize=(3.5 * len(weights) + 1, 3.4),
        sharey=True,
        layout="constrained",
    )
    for i, ax in enumerate(np.atleast_1d(axes)):
        z = values[:, i].reshape(current.shape)
        z[~valid] = np.nan
        im = ax.pcolormesh(
            current, mean, z, cmap="cividis", vmin=0, vmax=1, shading="auto"
        )
        ax.set(
            xlabel="Latest normalized rainfall",
            title=labels[i],
            xlim=(0, 1),
            ylim=(0, 1),
        )
    axes[0].set_ylabel("Seven-day mean normalized rainfall")
    fig.colorbar(im, ax=list(axes), label="Allocation weight", shrink=0.85)
    save(fig, out, "allocation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/dual")
    parser.add_argument("--out", default="runs/dual/figures")
    args = parser.parse_args()
    torch.set_num_threads(2)
    make_figures(args.run, args.out)


if __name__ == "__main__":
    main()
