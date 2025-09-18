from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from pandas.util import hash_pandas_object

import numpy as np
import pandas as pd


def build_user_features(
    events: pd.DataFrame,
    orders: pd.DataFrame,
    users: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Leakage-safe user features as of cutoff.

    Features: RFM (recency days, frequency purchases, monetary total), tenure days, country/device/channel.
    """
    users = users.copy()
    events = events.copy()
    orders = orders.copy()
    cutoff = pd.to_datetime(cutoff, utc=True)
    users["signup_ts"] = pd.to_datetime(users["signup_ts"], utc=True)
    events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True)
    orders["order_ts"] = pd.to_datetime(orders["order_ts"], utc=True)

    events_hist = events[events["event_ts"] <= cutoff]
    orders_hist = orders[orders["order_ts"] <= cutoff]

    # Recency from last event, frequency from orders count, monetary from orders sum
    last_event = events_hist.groupby("user_id")["event_ts"].max().rename("last_event_ts")
    recency_days = ((cutoff - last_event).dt.total_seconds() / 86400.0).rename("recency_days")

    freq = orders_hist.groupby("user_id")["order_id"].nunique().rename("freq")
    monetary = orders_hist.groupby("user_id")["amount"].sum().rename("monetary")

    tenure_days = ((cutoff - users.set_index("user_id")["signup_ts"]) / np.timedelta64(1, "D")).rename(
        "tenure_days"
    )

    feats = (
        users.set_index("user_id")[["country", "device", "acq_channel"]]
        .join([recency_days, freq, monetary, tenure_days])
        .fillna({"recency_days": 1e6, "freq": 0, "monetary": 0})
        .reset_index()
    )
    return feats


def split_time_based(
    df: pd.DataFrame, date_col: str, train_end: pd.Timestamp, valid_end: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.to_datetime(train_end, utc=True)
    valid_end = pd.to_datetime(valid_end, utc=True)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    train = df[df[date_col] <= train_end]
    valid = df[(df[date_col] > train_end) & (df[date_col] <= valid_end)]
    holdout = df[df[date_col] > valid_end]
    return train, valid, holdout


def make_experiment_dataset(
    users: pd.DataFrame,
    events: pd.DataFrame,
    orders: pd.DataFrame,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build a per-user dataset for uplift during an experiment window.

    Label = whether user purchased in [window_start, window_end]. Treatment from orders.treatment_flag max.
    """
    ws = pd.to_datetime(window_start, utc=True)
    we = pd.to_datetime(window_end, utc=True)
    orders_w = orders[(pd.to_datetime(orders["order_ts"], utc=True) >= ws) & (pd.to_datetime(orders["order_ts"], utc=True) <= we)]
    # Label: any purchase in window
    label = orders_w.groupby("user_id")["order_id"].nunique().gt(0).astype(int).rename("converted")
    # Treatment assignment: from any order in window if present; otherwise assign deterministically 50/50 among eligible
    treat_from_orders = orders_w.groupby("user_id")["treatment_flag"].max().rename("treatment_flag")
    eligible_ids = users.loc[pd.to_datetime(users["signup_ts"], utc=True) <= we, "user_id"].drop_duplicates()
    missing_ids = eligible_ids[~eligible_ids.isin(treat_from_orders.index)]
    if len(missing_ids) > 0:
        hashed = (hash_pandas_object(missing_ids, index=False) % 2).astype(int)
        extra = pd.Series(hashed.values, index=missing_ids.values, name="treatment_flag")
        treat_all = pd.concat([treat_from_orders, extra])
    else:
        treat_all = treat_from_orders
    # Build features at window start to avoid leakage
    feats = build_user_features(events, orders, users, cutoff=ws)
    out = (
        feats.merge(treat_all.rename_axis("user_id").reset_index(), on="user_id", how="left")
        .merge(label.rename_axis("user_id").reset_index(), on="user_id", how="left")
    )
    out["treatment_flag"] = out["treatment_flag"].fillna(0).astype(int)
    out["converted"] = out["converted"].fillna(0).astype(int)
    return out
