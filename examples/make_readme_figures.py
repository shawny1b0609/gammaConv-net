"""Draw the README figures from the public figure tables; no training is needed."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "figure_data"
OUT = ROOT / "docs" / "figures"
INK = "#243746"
GRAY = "#637482"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
RED = "#D55E00"
PINK = "#CC79A7"
MODELS = {"Single": BLUE, "Dual": GREEN, "Triple": RED, "XAJ": "#707780", "MLP": PINK}
PARTS = {"Aggregate": BLUE, "Fast": ORANGE, "Intermediate": GREEN, "Slow": BLUE}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.edgecolor": "#BCC7CE",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "gammaconv-figures",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    svg = OUT / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None})
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    print(name)


def heading(fig, title, subtitle):
    fig.text(0.055, 0.96, title, fontsize=21, weight="bold", va="top")
    fig.text(0.055, 0.895, subtitle, fontsize=12, color=GRAY, va="top")


def clean(ax, grid=True):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis="y", color="#E5EBEF", lw=0.8)
    ax.tick_params(length=3)


def box(ax, x, y, w, h, fill="white", edge="#D6E0E6", radius=0.12, lw=1):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, size=13, color=INK, weight="normal", **kwargs):
    ax.text(x, y, text, fontsize=size, color=color, weight=weight,
            ha=kwargs.pop("ha", "center"), va=kwargs.pop("va", "center"), **kwargs)


def arrow(ax, points, color=GRAY, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.4,
                ls="--" if dashed else "-", solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=12, linewidth=1.4, color=color,
                                linestyle="--" if dashed else "-", shrinkA=0, shrinkB=0))


def mini(fig, x, y, w, h, width, height):
    ax = fig.add_axes([x / width, y / height, w / width, h / height])
    ax.patch.set_alpha(0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    return ax


def architecture():
    """English redraw of manuscript Figure 1, with the released code's indexing."""
    width, height = 14, 10.6
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    label(ax, 0.5, 10.12, "GammaConv-Net", 25, weight="bold", ha="left")
    label(ax, 0.5, 9.72, "Prior-guided rainfall response + learned temporal features", 14, GRAY, ha="left")

    data = pd.read_csv(ROOT / "data" / "series_a.csv").iloc[2200:2310]
    for x, column, title, color in [(0.5, "rain", "Rainfall history", BLUE),
                                     (7.2, "flow", "Observed flow history", GREEN)]:
        box(ax, x, 7.05, 6.3, 2.3, "#F6F9FB")
        label(ax, x + 0.25, 9.0, title, 16, color, "bold", ha="left")
        plot = mini(fig, x + 0.3, 7.95, 5.7, 0.7, width, height)
        days = np.arange(len(data))
        plot.axvspan(len(data) - 30, len(data) - 0.5, color=color, alpha=0.09)
        if column == "rain":
            plot.bar(days, data[column], color=color, width=0.8, alpha=0.8)
        else:
            plot.plot(days, data[column], color=color, lw=1.6)
        plot.set_xlim(0, len(data) - 0.5)
        plot.set_ylim(bottom=0)
        label(ax, x + 3.15, 7.6, "Normalize  /  take the latest 30 daily observations", 12)
        seq = r"$P_{t-30},\ldots,P_{t-1}$" if column == "rain" else r"$Q_{t-30},\ldots,Q_{t-1}$"
        label(ax, x + 3.15, 7.28, seq, 14, color)

    arrow(ax, [(3.65, 7.05), (3.65, 6.55), (2.55, 6.55), (2.55, 6.05)], BLUE)
    arrow(ax, [(3.65, 6.55), (7.0, 6.55), (7.0, 6.05)], BLUE)
    arrow(ax, [(10.35, 7.05), (10.35, 6.55), (11.45, 6.55), (11.45, 6.05)], GREEN)

    branches = [(0.5, BLUE, "#EDF5FA", "01  Rainfall response"),
                (4.95, ORANGE, "#FCF7EB", "02  Rainfall features"),
                (9.4, GREEN, "#EDF7F3", "03  Flow memory")]
    for x, color, fill, title in branches:
        box(ax, x, 3.15, 4.1, 2.9, fill, edge=fill)
        ax.add_patch(Rectangle((x, 5.93), 4.1, 0.06, color=color, lw=0))
        label(ax, x + 2.05, 5.62, title, 15, color, "bold")

    label(ax, 2.55, 5.15, "Learnable Gamma kernels", 14)
    plot = mini(fig, 1.0, 4.13, 3.1, 0.8, width, height)
    t = np.linspace(0.08, 20, 200)
    for a, b, color in [(0.8, 2.5, ORANGE), (2.0, 1.1, GREEN), (4.0, 0.55, BLUE)]:
        g = t ** (a - 1) * np.exp(-b * t)
        plot.plot(t, g / g.max(), color=color, lw=2)
    plot.set_xlim(0, 20)
    plot.set_ylim(0, 1.08)
    label(ax, 2.55, 3.85, "Causal convolution", 13)
    label(ax, 2.55, 3.48, r"Rainfall-response feature  $q_{base}$", 13, BLUE)

    label(ax, 7.0, 5.15, "Linear layer + ReLU", 14)
    for row in range(3):
        for col in range(8):
            alpha = 0.15 + 0.07 * ((row * 3 + col * 2) % 9)
            ax.add_patch(Rectangle((5.59 + col * 0.36, 4.23 + row * 0.2), 0.27, 0.14,
                                   facecolor=ORANGE, alpha=alpha, lw=0))
    label(ax, 7.0, 3.85, "Mean over the input window", 13)
    label(ax, 7.0, 3.48, "Rainfall feature vector", 13, "#946500")

    label(ax, 11.45, 5.15, "LSTM encoder", 14)
    for x, text in [(9.92, r"$h_1$"), (11.0, r"$\cdots$"), (12.08, r"$h_{30}$")]:
        box(ax, x, 4.25, 0.83, 0.48, "white", "#A8D6C5", radius=0.06)
        label(ax, x + 0.415, 4.49, text, 14, GREEN)
    arrow(ax, [(10.75, 4.49), (11.0, 4.49)], GREEN)
    arrow(ax, [(11.83, 4.49), (12.08, 4.49)], GREEN)
    label(ax, 11.45, 3.85, "Last hidden state", 13)
    label(ax, 11.45, 3.48, "Temporal feature vector", 13, GREEN)

    arrow(ax, [(2.55, 3.15), (2.55, 2.85), (7.0, 2.85), (7.0, 2.65)], BLUE)
    arrow(ax, [(11.45, 3.15), (11.45, 2.85), (7.0, 2.85), (7.0, 2.65)], GREEN)
    arrow(ax, [(7.0, 3.15), (7.0, 2.65)], "#B78413")
    box(ax, 4.7, 2.1, 4.6, 0.55, INK, INK)
    label(ax, 7.0, 2.375, "Concatenate features", 15, "white", "bold")
    arrow(ax, [(7.0, 2.1), (7.0, 1.7)])
    box(ax, 4.7, 0.95, 4.6, 0.75, "#F0F3F5")
    label(ax, 7.0, 1.325, "MLP fusion + Softplus output", 14, weight="bold")

    label(ax, 2.55, 1.48, "30 observed days", 13, GRAY)
    label(ax, 2.55, 1.12, "1-day-ahead prediction", 13, GRAY)
    arrow(ax, [(9.3, 1.325), (10.0, 1.325)])
    box(ax, 10.0, 0.88, 3.5, 0.9, "#EDF5FA", "#B8D9EA")
    label(ax, 11.75, 1.48, "Inverse flow scaling", 12, GRAY)
    label(ax, 11.75, 1.12, r"Next-day flow  $\hat{Q}_{t}$", 16, BLUE, "bold")
    label(ax, 0.5, 0.36, "The dual model also fuses allocation weights. Kernel shapes are schematic.", 11, GRAY, ha="left")
    save(fig, "architecture")


def variants():
    """Compare the three kernel paths, following manuscript Figure 2."""
    width, height = 14, 7.3
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    label(ax, 0.5, 6.91, "Single-, dual-, and triple-kernel structures", 22, weight="bold", ha="left")
    label(ax, 0.5, 6.47, "Rainfall allocation and Gamma convolution", 13, GRAY, ha="left")
    rain = pd.read_csv(ROOT / "data" / "series_a.csv")["rain"].iloc[2271:2301].to_numpy()
    for left, name, parts in [(0.5, "Single", ["Aggregate"]),
                              (5.0, "Dual", ["Fast", "Slow"]),
                              (9.5, "Triple", ["Fast", "Intermediate", "Slow"])]:
        w = 4.0
        color = MODELS[name]
        box(ax, left, 0.8, w, 5.2, "#F7F9FA", "#E3EAEE")
        label(ax, left + w / 2, 5.64, name + " kernel", 17, color, "bold")
        plot = mini(fig, left + 0.55, 4.76, 2.9, 0.52, width, height)
        plot.bar(np.arange(30), rain, color=BLUE, width=0.8)
        plot.set_xlim(-1, 30)
        label(ax, left + 2, 4.49, "Rainfall window", 12)
        if len(parts) > 1:
            arrow(ax, [(left + 2, 4.28), (left + 2, 4.04)])
            box(ax, left + 0.35, 3.40, 3.3, 0.64, "#F1EAF3", "#E2CFE5", radius=0.08)
            label(ax, left + 2, 3.81, "Rainfall allocation", 13, "#74516D", "bold")
            label(ax, left + 2, 3.55, "Latest rain + 7-day mean", 10.7, "#74516D")
        else:
            arrow(ax, [(left + 2, 4.28), (left + 2, 2.95)], BLUE)
            label(ax, left + 2.12, 3.68, "All rainfall", 12, GRAY, ha="left")

        centers = np.linspace(left + 0.7, left + 3.3, len(parts)) if len(parts) > 1 else [left + 2]
        if len(parts) > 1:
            for center in centers:
                arrow(ax, [(left + 2, 3.40), (left + 2, 3.17), (center, 3.17), (center, 2.95)])
        for center, part in zip(centers, parts):
            plot = mini(fig, center - 0.5, 2.03, 1.0, 0.75, width, height)
            t = np.linspace(0.08, 24, 200)
            a, b = {"Aggregate": (1.7, 1.3), "Fast": (0.7, 3.1),
                    "Intermediate": (1.6, 1.3), "Slow": (3.9, 0.45)}[part]
            g = t ** (a - 1) * np.exp(-b * t)
            plot.fill_between(t, g / g.max(), color=PARTS[part], alpha=0.12)
            plot.plot(t, g / g.max(), color=PARTS[part], lw=2.0)
            plot.set_xlim(0, 24)
            label(ax, center, 1.81, part, 10.5 if part == "Intermediate" else 12, PARTS[part])
            if len(parts) > 1:
                arrow(ax, [(center, 1.57), (center, 1.39), (left + 2, 1.39), (left + 2, 1.25)], PARTS[part])
            else:
                arrow(ax, [(center, 1.57), (center, 1.25)], PARTS[part])
        label(ax, left + 2, 1.05, r"$q_{base}$" if len(parts) == 1 else r"Sum responses  $\rightarrow q_{base}$", 14, weight="bold")
    label(ax, 0.5, 0.36, "Allocation is fixed within each window. Response shapes are schematic.", 11, GRAY, ha="left")
    save(fig, "kernel_variants")


def paper_performance():
    data = pd.read_csv(DATA / "paper_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.16, top=0.68, wspace=0.29)
    heading(fig, "Performance across training fractions", "Reported in the manuscript, Table 1  /  five models, six sample sizes")
    ticks = [5, 10, 25, 50, 75, 100]
    for ax, key, title in zip(axes, ["nse", "rmse", "mae"],
                              ["NSE  ↑", "RMSE  ↓  (m³/s)", "MAE  ↓  (m³/s)"]):
        for name, color in MODELS.items():
            rows = data[data.model == name].sort_values("fraction_pct")
            style = {"XAJ": "--", "MLP": ":"}.get(name, "-")
            marker = {"Single": "o", "Dual": "s", "Triple": "^", "XAJ": "D", "MLP": "v"}[name]
            ax.plot(np.arange(6), rows[key], marker=marker, color=color, label=name,
                    ls=style, lw=1.8, markersize=5)
        ax.set_xticks(np.arange(6), ticks)
        ax.set_xlabel("Training fraction (%)")
        ax.set_title(title, loc="left")
        clean(ax)
    axes[0].set_ylim(0.63, 0.94)
    axes[0].set_yticks([0.65, 0.75, 0.85, 0.90])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.05, 0.82), ncol=5, frameon=False,
               columnspacing=2.0)
    fig.text(0.065, 0.035, "Source: Yang et al. (2026), Table 1. Training fractions are shown as six ordered settings.", fontsize=11, color=GRAY)
    save(fig, "paper_performance")


def sampling():
    data = pd.read_csv(ROOT / "data" / "series_a.csv")
    blocks = pd.read_csv(DATA / "sampling.csv")
    split = 3286
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2]})
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.80, hspace=0.26)
    heading(fig, "Small-sample training blocks", "De-identified series A  /  contiguous blocks within the chronological training prefix")
    for ax in axes:
        ax.axvspan(split + 0.5, len(data) + 0.5, color="#E6ECF0", zorder=0)
        ax.axvline(split + 0.5, ls="--", color=GRAY, lw=1)
        ax.set_xlim(1, len(data))
        clean(ax)
    axes[0].plot(data.day, data.flow, color=BLUE, lw=0.9)
    axes[0].set_ylabel("Flow (m³/s)")
    axes[0].text(0.02, 0.9, "Training prefix: 3,286 days", transform=axes[0].transAxes, fontsize=12)
    axes[0].text(0.835, 0.9, "Validation", transform=axes[0].transAxes, fontsize=12, color=GRAY)
    for i, row in enumerate(blocks.itertuples()):
        axes[1].barh(i, row.days, left=row.first_day - 0.5, height=0.50,
                     color=GREEN if row.fraction_pct == 5 else BLUE, alpha=1 if row.fraction_pct == 5 else 0.63)
        axes[1].text(row.first_day + row.days / 2, i, f"{row.days:,}",
                     ha="center", va="center", color="white", fontsize=10,
                     weight="bold" if row.fraction_pct == 5 else "normal")
    axes[1].set_yticks(range(6), [f"{v}%" for v in blocks.fraction_pct])
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Training fraction")
    axes[1].set_xlabel("Anonymous day index")
    axes[1].set_xticks([1, 1000, 2000, 3000, 4018])
    axes[1].grid(False)
    fig.text(0.09, 0.035, "Numbers inside bars are block lengths. Selection and scaling use the full training prefix.", fontsize=11, color=GRAY)
    save(fig, "sampling")


def validation():
    data = pd.read_csv(DATA / "example_predictions.csv")
    records = json.loads((ROOT / "docs" / "example_runs.json").read_text(encoding="utf-8"))
    peak = int(data.loc[data.observed.idxmax(), "day"])
    zoom = data[data.day.between(peak - 15, peak + 15)]
    fig, axes = plt.subplots(3, 2, figsize=(14, 8.2), gridspec_kw={"width_ratios": [2.7, 1]})
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.76, hspace=0.32, wspace=0.15)
    heading(fig, "Validation hydrographs", "Release example  /  5% training block, 150 epochs, seed 42  /  de-identified series A")
    handles = [Line2D([], [], color=INK, lw=1.3, label="Observed")]
    handles += [Line2D([], [], color=MODELS[name], lw=1.7, label=name) for name in ["Single", "Dual", "Triple"]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.84), ncol=4, frameon=False)
    for i, name in enumerate(["Single", "Dual", "Triple"]):
        model = name.lower()
        for ax, frame in zip(axes[i], [data, zoom]):
            ax.plot(frame.day, frame.observed, color=INK, lw=1.2)
            ax.plot(frame.day, frame[model], color=MODELS[name], lw=1.4, alpha=0.9)
            ax.set_ylim(0, data.observed.max() * 1.12)
            clean(ax)
            if i < 2:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Anonymous day index")
        nse = records[model]["validation"]["nse"]
        axes[i, 0].text(0.32, 0.84, f"{name}  ·  NSE {nse:.3f}",
                        transform=axes[i, 0].transAxes, fontsize=12, weight="bold", color=MODELS[name])
        axes[i, 0].axvspan(zoom.day.min(), zoom.day.max(), color=GRAY, alpha=0.10)
        axes[i, 0].set_xlim(data.day.min(), data.day.max())
        axes[i, 1].set_xlim(zoom.day.min(), zoom.day.max())
        axes[i, 1].tick_params(labelleft=False)
    axes[0, 1].set_title("Largest observed peak", fontsize=12, loc="left", pad=10)
    fig.supylabel("Flow (m³/s)", x=0.018, y=0.44, fontsize=13)
    fig.text(0.08, 0.035, "All models use the same 702 validation targets. Checkpoints are selected by validation RMSE.", fontsize=11, color=GRAY)
    save(fig, "validation_models")


def kernels():
    data = pd.read_csv(DATA / "example_kernels.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.72, wspace=0.17)
    heading(fig, "Learned daily response kernels", "Discrete weights from the released example checkpoints  /  lags 0–29 days")
    for ax, name in zip(axes, ["Single", "Dual", "Triple"]):
        rows = data[data.model == name.lower()]
        for part in PARTS:
            frame = rows[rows.component == part]
            if frame.empty:
                continue
            ax.plot(frame.lag, frame.weight, color=PARTS[part], lw=1.8, marker="o",
                    ms=3, label=part)
        ax.set_title(name + " kernel", loc="left", color=MODELS[name])
        ax.set_xlabel("Lag (days)")
        ax.set_xticks([0, 7, 14, 21, 29])
        ax.set_xlim(-0.7, 29.5)
        ax.set_ylim(-0.03, 1.05)
        ax.legend(frameon=False, fontsize=11, loc="upper right")
        clean(ax)
    axes[0].set_ylabel("Normalized response weight")
    fig.text(0.07, 0.035, "Each component kernel is normalized to sum to one.", fontsize=11, color=GRAY)
    save(fig, "model_kernels")


def allocation():
    data = pd.read_csv(DATA / "example_allocation.csv")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.14, top=0.78, wspace=0.19, hspace=0.41)
    heading(fig, "Rainfall allocation", "Learned weights from the dual- and triple-kernel example checkpoints")
    cmap = LinearSegmentedColormap.from_list("allocation", ["#F4F8FA", "#B8DDEC", BLUE, "#183D55"])
    grid = np.linspace(0, 1, 41)
    panels = [(0, 0, "dual", "Fast"), (0, 1, "dual", "Slow"),
              (1, 0, "triple", "Fast"), (1, 1, "triple", "Intermediate"), (1, 2, "triple", "Slow")]
    for row, col, model, part in panels:
        ax = axes[row, col]
        frame = data[(data.model == model) & (data.component == part)]
        values = np.full((41, 41), np.nan)
        x = np.rint(frame.latest.to_numpy() * 40).astype(int)
        y = np.rint(frame["mean"].to_numpy() * 40).astype(int)
        values[y, x] = frame.weight.to_numpy()
        mesh = ax.pcolormesh(grid, grid, np.ma.masked_invalid(values), shading="nearest",
                             cmap=cmap, vmin=0, vmax=1, rasterized=True)
        ax.set_title(f"{model.title()} · {part}", color=PARTS[part], loc="left", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_aspect("equal")
        ax.tick_params(labelbottom=True)
        if col == 0:
            ax.set_ylabel("7-day mean rainfall")
        ax.set_xlabel("Latest rainfall")
    axes[0, 2].axis("off")
    axes[0, 2].text(0.03, 0.76, "Within each window", fontsize=14, weight="bold", transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.03, 0.57, "Allocation weights sum to 1.\n\nInputs are normalized rainfall.\n\nBlank cells are infeasible\ninput combinations.",
                     fontsize=12, color=GRAY, va="top", transform=axes[0, 2].transAxes, linespacing=1.3)
    cax = fig.add_axes([0.92, 0.22, 0.015, 0.49])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label("Allocation weight", labelpad=10)
    fig.text(0.08, 0.035, "The 7-day mean includes the latest value. Feasibility assumes each of the seven normalized values lies in [0, 1].", fontsize=11, color=GRAY)
    save(fig, "model_allocation")


def response_times():
    data = pd.read_csv(DATA / "paper_response_times.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.17, top=0.72, wspace=0.26)
    heading(fig, "Response time scales", "Continuous Gamma means (α / β), transcribed from manuscript Table 2")
    for ax, name in zip(axes, ["Single", "Dual", "Triple"]):
        rows = data[data.model == name]
        for part in PARTS:
            frame = rows[rows.component == part].sort_values("fraction_pct")
            if frame.empty:
                continue
            ax.plot(np.arange(6), frame.mean_days, color=PARTS[part], marker="o", lw=1.8, ms=5, label=part)
        ax.set_xticks(np.arange(6), [5, 10, 25, 50, 75, 100])
        ax.set_xlabel("Training fraction (%)")
        ax.set_title(name + " kernel", loc="left", color=MODELS[name])
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, loc="best", fontsize=11)
        clean(ax)
    axes[0].set_ylabel("Mean response time (days)")
    fig.text(0.065, 0.035, "Source: Yang et al. (2026), Table 2. Panels use different vertical scales.", fontsize=11, color=GRAY)
    save(fig, "paper_response_times")


if __name__ == "__main__":
    architecture()
    variants()
    paper_performance()
    sampling()
    validation()
    kernels()
    allocation()
    response_times()
