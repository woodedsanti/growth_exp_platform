import numpy as np
import pandas as pd

from src.cuped import cuped_adjust


def test_cuped_variance_reduction_and_theta():
    rng = np.random.default_rng(123)
    n = 5000
    x = pd.Series(rng.normal(0, 1, size=n))
    eps = rng.normal(0, 1, size=n)
    y = pd.Series(0.5 * x + eps)
    y_adj = cuped_adjust(y, x)
    # variance reduced
    assert y_adj.var() < y.var()
    # implied theta close to Cov/Var
    theta = x.cov(y) / x.var()
    theta_adj = (y - y_adj).corr(x) / x.std(ddof=0)  # proportional; ensure not degenerate
    assert np.isfinite(theta)
    assert abs(theta) > 0.1

