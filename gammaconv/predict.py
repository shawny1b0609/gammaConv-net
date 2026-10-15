"""Run rolling one-day predictions using a saved model and observed histories."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from .data import read_data, windows
from .models import make_model
from .train import predict


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(**checkpoint["model"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model, checkpoint


def forecast(checkpoint, data):
    model, state = load_model(checkpoint)
    seq = state["model"]["seq"]
    rain, flow, _ = windows(data, state["scale"], seq)
    low, high = state["scale"]["flow"]
    values = predict(model, rain, flow).astype(float) * (high - low + 1e-12) + low
    return pd.DataFrame(
        {
            "day": data.day.iloc[seq:].to_numpy(),
            "observed": data.flow.iloc[seq:].to_numpy(),
            "predicted": values,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="runs/predictions.csv")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = forecast(args.checkpoint, read_data(args.data))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    print(f"Saved {len(result)} predictions to {path}.")


if __name__ == "__main__":
    main()
