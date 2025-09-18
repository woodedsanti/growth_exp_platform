from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def mape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def wmape(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    weight: str | pd.Series | np.ndarray = "amount",
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if isinstance(weight, str):
        # weight is name in y_true/y_pred df context; not used here
        w = np.ones_like(y_true)
    else:
        w = np.asarray(weight, dtype=float)
    denom = np.sum(np.abs(y_true) * w)
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred) * w) / denom


def diff_in_means(
    y: pd.Series | np.ndarray, treat: pd.Series | np.ndarray, alpha: float = 0.05
) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    t = np.asarray(treat, dtype=int)
    y1, y0 = y[t == 1], y[t == 0]
    n1, n0 = len(y1), len(y0)
    m1, m0 = y1.mean(), y0.mean()
    v1, v0 = y1.var(ddof=1), y0.var(ddof=1)
    effect = m1 - m0
    se = np.sqrt(v1 / n1 + v0 / n0)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = effect - z * se, effect + z * se
    return {"effect": float(effect), "se": float(se), "ci_low": float(ci_low), "ci_high": float(ci_high)}


def diff_in_proportions(
    success: pd.Series | np.ndarray, treat: pd.Series | np.ndarray, alpha: float = 0.05
) -> Dict[str, float]:
    y = np.asarray(success, dtype=int)
    t = np.asarray(treat, dtype=int)
    y1, y0 = y[t == 1], y[t == 0]
    n1, n0 = len(y1), len(y0)
    p1, p0 = y1.mean(), y0.mean()
    effect = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = effect - z * se, effect + z * se
    return {"effect": float(effect), "se": float(se), "ci_low": float(ci_low), "ci_high": float(ci_high)}

