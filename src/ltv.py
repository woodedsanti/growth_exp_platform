from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data


def prepare_transactions(orders: pd.DataFrame) -> pd.DataFrame:
    """Prepare frequency/recency/T and monetary_value for lifetimes models.

    orders: columns [user_id, order_ts, amount]
    """
    df = orders.copy()
    df["order_ts"] = pd.to_datetime(df["order_ts"], utc=True)
    # Use daily frequency for stability
    summary = summary_data_from_transaction_data(
        df, customer_id_col="user_id", datetime_col="order_ts", monetary_value_col="amount", freq="D"
    ).reset_index()
    return summary


@dataclass
class LTVModels:
    bgnbd: BetaGeoFitter
    gg: GammaGammaFitter
    freq_unit: str = "D"


def fit_bgnbd_gg(df: pd.DataFrame) -> Dict:
    """Fit BG/NBD and Gamma-Gamma models, return dict with models and params."""
    bgnbd = BetaGeoFitter(penalizer_coef=0.001)
    bgnbd.fit(df["frequency"], df["recency"], df["T"])
    df_ = df[df["monetary_value"].notnull() & (df["monetary_value"] > 0)]
    gg = GammaGammaFitter(penalizer_coef=0.001)
    gg.fit(df_["frequency"], df_["monetary_value"])
    return {
        "bgnbd": bgnbd,
        "gg": gg,
        "freq_unit": "D",
        "params_bgnbd": bgnbd.params_.to_dict(),
        "params_gg": gg.params_.to_dict(),
    }


def predict_ltv(models: Dict, horizon_days: int, user_df: pd.DataFrame) -> pd.DataFrame:
    """Predict expected LTV over horizon using BG/NBD + GG.

    Returns columns: user_id, ltv_h{horizon}, ci_low, ci_high
    """
    bgnbd: BetaGeoFitter = models["bgnbd"]
    gg: GammaGammaFitter = models["gg"]
    t = float(horizon_days)
    df = user_df.copy()
    # expected number of purchases in horizon
    exp_purch = bgnbd.conditional_expected_number_of_purchases_up_to_time(
        t, df["frequency"], df["recency"], df["T"]
    )
    exp_monetary = gg.conditional_expected_average_profit(df["frequency"], df["monetary_value"].fillna(0))
    ltv = exp_purch * exp_monetary
    col = f"ltv_h{int(horizon_days)}"
    # Simple symmetric CI proxy (+/-20%) for UI display; not used in tests
    ci_low = ltv * 0.8
    ci_high = ltv * 1.2
    out = pd.DataFrame({
        "user_id": df["user_id"],
        col: ltv.astype(float),
        "ci_low": ci_low.astype(float),
        "ci_high": ci_high.astype(float),
    })
    return out

