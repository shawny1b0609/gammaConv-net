"""Run the original six-fraction, three-model small-sample experiment."""

import argparse
from pathlib import Path
import pandas as pd

from .data import read_data
from .train import Config, run


def settings(n):
    """Original sample-count-dependent optimizer and model settings."""
    rows = [
        (150, 8e-4, 2500, 5e-4, 300, 25, 18, 12, 64),
        (300, 1e-3, 2000, 1e-4, 250, 20, 15, 12, 72),
        (700, 1.2e-3, 1500, 8e-5, 200, 15, 12, 10, 80),
        (1500, 1.5e-3, 1200, 5e-5, 180, 12, 10, 8, 88),
        (2500, 1.5e-3, 1000, 3e-5, 150, 10, 8, 6, 96),
        (float("inf"), 2e-3, 800, 1e-5, 120, 8, 6, 5, 112),
    ]
    for limit, lr, epochs, wd, patience, peak, low, mid, hidden in rows:
        if n < limit:
            return dict(
                lr=lr,
                epochs=epochs,
                wd=wd,
                patience=patience,
                peak=peak,
                low=low,
                mid=mid,
                hidden=hidden,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/series_a.csv")
    parser.add_argument("--out", default="runs/experiment")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.25, 0.50, 0.75, 1.0],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["single", "dual", "triple"],
        default=["single", "dual", "triple"],
    )
    parser.add_argument(
        "--epochs", type=int, help="Override every preset for a shorter run."
    )
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = read_data(args.data)
    rows = []
    for fraction in args.fractions:
        hp = settings(int(int(len(data) * 0.818) * fraction) - 30)
        if args.epochs is not None:
            hp["epochs"] = args.epochs
        for name in args.models:
            current = dict(hp)
            current["hidden"] = int(
                hp["hidden"] * {"single": 1.0, "dual": 1.2, "triple": 1.4}[name]
            )
            folder = Path(args.out) / f"{fraction:g}" / name
            record = run(
                Config(
                    data=args.data,
                    out=str(folder),
                    model=name,
                    fraction=fraction,
                    loss="bands",
                    device=args.device,
                    seed=args.seed,
                    **current,
                )
            )
            rows.append(
                {
                    "fraction": fraction,
                    "model": name,
                    "best_epoch": record["best_epoch"],
                    **record["validation"],
                }
            )
            pd.DataFrame(rows).to_csv(Path(args.out) / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
