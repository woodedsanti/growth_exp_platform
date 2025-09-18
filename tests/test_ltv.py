import numpy as np
import pandas as pd

from src.ltv import fit_bgnbd_gg, prepare_transactions, predict_ltv


def test_ltv_monotonic():
    # toy transactions
    orders = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3", "o4", "o5"],
            "user_id": ["u1", "u1", "u2", "u2", "u2"],
            "order_ts": [
                "2023-01-01T00:00:00Z",
                "2023-03-01T00:00:00Z",
                "2023-01-05T00:00:00Z",
                "2023-02-05T00:00:00Z",
                "2023-03-05T00:00:00Z",
            ],
            "amount": [10.0, 10.0, 20.0, 20.0, 20.0],
            "margin": [6, 6, 12, 12, 12],
            "promo_flag": [0, 0, 0, 0, 0],
            "treatment_flag": [0, 0, 0, 0, 0],
        }
    )
    tx = prepare_transactions(orders)
    models = fit_bgnbd_gg(tx)
    preds = predict_ltv(models, 365, tx.merge(orders[["user_id"]].drop_duplicates(), on="user_id"))
    p = preds.set_index("user_id").iloc[:, 0]
    assert p["u2"] > p["u1"]  # more frequent and higher spend → higher LTV

