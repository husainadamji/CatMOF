"""Regression metrics for reporting."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray):
    """Return MAE, R², MAPE, RMSE. MAPE can be unstable near zero targets."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return mae, r2, mape, rmse
