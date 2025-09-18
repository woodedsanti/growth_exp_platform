from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px


def make_cohorts(
    events: pd.DataFrame, users: pd.DataFrame, freq: Literal["W", "M"] = "W"
) -> pd.DataFrame:
    """Build cohort retention table at weekly or monthly granularity.

    A user is retained in period k if they have >=1 session in that k-th period after signup.

    Returns DataFrame indexed by [cohort, period] with columns [n_users, retained_users, retention_rate].
    """
    users = users.copy()
    events = events.copy()
    users["signup_ts"] = pd.to_datetime(users["signup_ts"], utc=True)
    events["event_ts"] = pd.to_datetime(events["event_ts"], utc=True)

    # Use all events as activity; if event_type present, any event counts
    df = events.merge(users[["user_id", "signup_ts"]], on="user_id", how="inner", validate="m:1")
    # Non-negative periods only
    delta = (df["event_ts"] - df["signup_ts"]).dt.total_seconds()
    step = 7 * 24 * 3600 if freq == "W" else 30 * 24 * 3600
    # Safe conversion: drop NaNs before casting to int and filter non-negative periods
    period = np.floor(delta / step)
    df = df.assign(period=pd.to_numeric(period, errors="coerce"))
    df = df[df["period"].notna()]
    df["period"] = df["period"].astype(int)
    df = df[df["period"] >= 0]

    # Cohort key is signup week/month start
    if freq == "W":
        # Use Monday-start weeks via Period, then localize to UTC to keep tz-aware
        ck = users["signup_ts"].dt.tz_convert("UTC").dt.to_period("W-MON").dt.start_time
        cohort_key = ck.dt.tz_localize("UTC")
    else:
        ck = users["signup_ts"].dt.tz_convert("UTC").dt.to_period("M").dt.start_time
        cohort_key = ck.dt.tz_localize("UTC")
    users = users.assign(cohort=cohort_key.values)

    df = df.merge(users[["user_id", "cohort"]], on="user_id", how="left")
    # Unique users per cohort
    cohort_sizes = users.groupby("cohort")["user_id"].nunique().rename("n_users")
    # Retained users per cohort-period
    retained = df.groupby(["cohort", "period"])['user_id'].nunique().rename("retained_users")
    res = retained.reset_index().merge(cohort_sizes.reset_index(), on="cohort", how="left")
    res["retention_rate"] = (res["retained_users"] / res["n_users"]).fillna(0.0)
    res = res.set_index(["cohort", "period"]).sort_index()
    return res


def plot_retention_heatmap(cohort_df: pd.DataFrame):
    """Plot retention heatmap using Plotly.

    Expects DataFrame with index [cohort, period] and column retention_rate.
    """
    df = (
        cohort_df.reset_index()
        .pivot(index="cohort", columns="period", values="retention_rate")
        .sort_index()
    )
    fig = px.imshow(
        df,
        color_continuous_scale="Blues",
        aspect="auto",
        origin="lower",
        labels=dict(color="Retention"),
    )
    fig.update_layout(margin=dict(l=40, r=10, t=30, b=40))
    return fig
