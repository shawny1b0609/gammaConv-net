"""Validate daily data and build chronological one-step windows."""

import numpy as np
import pandas as pd
import torch


def read_data(path):
    """Read day, flow, rain without dropping or filling observations."""
    data = pd.read_csv(path)
    if list(data.columns) != ["day", "flow", "rain"]:
        raise ValueError("CSV columns must be day,flow,rain in that order.")
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="raise")
    if len(data) < 2 or not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Data must contain at least two rows and only finite values.")
    days = data.day.to_numpy(dtype=float)
    if (
        (days < 1).any()
        or (days != np.floor(days)).any()
        or not np.all(np.diff(days) == 1)
    ):
        raise ValueError("day must contain consecutive positive integers.")
    if (data[["flow", "rain"]] < 0).any().any():
        raise ValueError("Rainfall and flow must be nonnegative.")
    data["day"] = data.day.astype("int64")
    return data


def select_block(flow, fraction):
    """Select a contiguous high-flow block using the original scoring rule."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1].")
    size = int(len(flow) * fraction)
    if size < 1:
        raise ValueError("The selected training block is empty.")
    if fraction == 1:
        return 0, len(flow)
    best, start = -np.inf, 0
    for i in range(len(flow) - size + 1):
        block = flow[i : i + size]
        if fraction <= 0.10:
            score = 0.6 * block.max() + 0.4 * np.percentile(block, 90)
        elif fraction <= 0.25:
            score = 0.4 * block.max() + 0.3 * block.mean() + 0.3 * block.std()
        else:
            score = (
                0.3 * block.max()
                + 0.3 * block.mean()
                + 0.2 * block.std()
                + 0.2 * (block.max() - block.min())
            )
        if score > best:
            best, start = score, i
    return start, start + size


def fit_scale(data):
    """Fit min-max scaling to the chronological training prefix only."""
    return {
        col: [float(data[col].min()), float(data[col].max())]
        for col in ("rain", "flow")
    }


def scale_values(data, scale):
    return {
        col: (data[col].to_numpy(dtype=float) - scale[col][0])
        / (scale[col][1] - scale[col][0] + 1e-12)
        for col in ("rain", "flow")
    }


def windows(data, scale, seq):
    """Use days i..i+seq-1 to predict flow on day i+seq."""
    if seq < 1 or len(data) <= seq:
        raise ValueError("Each split needs more rows than the input window length.")
    values = scale_values(data, scale)
    n = len(data) - seq
    rain = np.stack([values["rain"][i : i + seq] for i in range(n)])[:, :, None]
    flow = np.stack([values["flow"][i : i + seq] for i in range(n)])[:, :, None]
    target = values["flow"][seq:, None]
    return tuple(torch.tensor(x, dtype=torch.float32) for x in (rain, flow, target))


def prepare(data, seq=30, split=0.818, fraction=1.0):
    """Split first, then select the training block and form windows separately."""
    if not 0 < split < 1:
        raise ValueError("split must lie in (0, 1).")
    cut = int(len(data) * split)
    train, valid = data.iloc[:cut], data.iloc[cut:]
    if min(len(train), len(valid)) <= seq:
        raise ValueError("Both chronological splits must be longer than seq.")
    start, end = select_block(train.flow.to_numpy(dtype=float), fraction)
    scale = fit_scale(train)
    block = train.iloc[start:end]
    info = {
        "rows": len(data),
        "split_row": cut,
        "train_start_day": int(block.day.iloc[0]),
        "train_end_day": int(block.day.iloc[-1]),
        "train_windows": len(block) - seq,
        "valid_windows": len(valid) - seq,
        "valid_first_target_day": int(valid.day.iloc[seq]),
        "scaling": "full chronological training prefix",
        "selection": "contiguous high-flow block" if fraction < 1 else "full prefix",
    }
    return windows(block, scale, seq), windows(valid, scale, seq), scale, info
