import numpy as np
import pandas as pd

from src.uplift import target_top_k_percent, uplift_qini_auc


def test_uplift_qini_better_than_random_with_true_scores():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(0, 1, size=n)
    t = rng.integers(0, 2, size=n)
    # true uplift proportional to x
    uplift_true = 0.2 * (x - x.mean())
    base = rng.normal(0, 1, size=n)
    y = (base + (t * uplift_true) + rng.normal(0, 0.5, size=n)) > 0.5
    y = y.astype(int)

    metrics_true = uplift_qini_auc(pd.Series(y), pd.Series(t), pd.Series(uplift_true))
    metrics_rand = uplift_qini_auc(pd.Series(y), pd.Series(t), pd.Series(rng.normal(0, 1, size=n)))
    assert metrics_true["qini_auc"] > metrics_rand["qini_auc"]


def test_target_top_k_percent():
    s = pd.Series([0.1, 0.2, 0.3, 0.4])
    mask = target_top_k_percent(s, 0.5)
    assert mask.sum() == 2
    assert mask.idxmax() == 3  # top score included

