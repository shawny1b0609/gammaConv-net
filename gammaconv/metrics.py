"""Metrics in the supplied flow units."""

import numpy as np


def scores(obs, pred):
    obs, pred = np.asarray(obs, dtype=float), np.asarray(pred, dtype=float)
    if obs.ndim != 1 or obs.shape != pred.shape or len(obs) == 0:
        raise ValueError("Observed and predicted flows must be equal-length vectors.")
    if not np.isfinite(obs).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require finite values.")
    error = pred - obs
    variance = np.sum((obs - obs.mean()) ** 2)
    return {
        "nse": float(1 - np.sum(error**2) / variance) if variance > 0 else None,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }
