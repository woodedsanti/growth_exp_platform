from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def cuped_adjust(y: pd.Series, x_cov: pd.Series) -> pd.Series:
    """Apply CUPED adjustment to outcome y using covariate x.

    theta = Cov(Y, X) / Var(X)
    Y_adj = Y - theta * (X - E[X])
    """
    y = pd.Series(y).astype(float)
    x = pd.Series(x_cov).astype(float)
    var_x = x.var(ddof=0)
    if var_x == 0 or np.isnan(var_x):
        return y.copy()
    theta = x.cov(y) / x.var()
    y_adj = y - theta * (x - x.mean())
    return y_adj


def _ate_and_se(y: pd.Series, t: pd.Series) -> Dict[str, float]:
    y1 = y[t == 1]
    y0 = y[t == 0]
    n1, n0 = len(y1), len(y0)
    m1, m0 = y1.mean(), y0.mean()
    v1, v0 = y1.var(ddof=1), y0.var(ddof=1)
    effect = m1 - m0
    se = float(np.sqrt(v1 / n1 + v0 / n0))
    return {"effect": float(effect), "se": se}


def cuped_effect_summary(
    y: pd.Series, x_cov: pd.Series, treat: pd.Series
) -> Dict[str, float]:
    """Compute ATE and SE before/after CUPED and variance reduction percent."""
    y = pd.Series(y).astype(float)
    x = pd.Series(x_cov).astype(float)
    t = pd.Series(treat).astype(int)

    base = _ate_and_se(y, t)
    y_adj = cuped_adjust(y, x)
    adj = _ate_and_se(y_adj, t)
    var_reduction = max(0.0, 1 - (y_adj.var() / y.var()) if y.var() > 0 else 0.0)
    return {
        "effect_before": base["effect"],
        "se_before": base["se"],
        "effect_after": adj["effect"],
        "se_after": adj["se"],
        "variance_reduction_pct": float(var_reduction * 100.0),
    }

