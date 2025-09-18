from __future__ import annotations

import argparse
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)


@dataclass
class SimConfig:
    n_users: int = 100_000
    months: int = 18
    start_date: datetime = datetime(2023, 1, 1, tzinfo=timezone.utc)
    experiment_days: int = 28
    experiment_start_offset_days: int = 400  # ensures inside the horizon


COUNTRIES = ["US", "MX", "CA", "GB", "DE"]
DEVICES = ["ios", "android", "web"]
CHANNELS = ["paid", "organic", "referral", "affiliate"]


def seasonal_daily_signups(days: int, base: float = 500.0) -> np.ndarray:
    t = np.arange(days)
    dow = t % 7
    # weekly seasonality (weekend dip) + annual seasonality
    weekly = 1.0 - 0.25 * ((dow == 5) | (dow == 6))
    annual = 1.0 + 0.2 * np.sin(2 * np.pi * t / 365)
    lam = base * weekly * annual
    return RNG.poisson(lam)


def simulate_users(cfg: SimConfig) -> pd.DataFrame:
    days = cfg.months * 30
    signups_per_day = seasonal_daily_signups(days)
    signup_dates = []
    for d, n in enumerate(signups_per_day):
        signup_dates.extend([cfg.start_date + timedelta(days=int(d))] * int(n))
    signup_dates = np.array(signup_dates, dtype="datetime64[ns]")
    if len(signup_dates) == 0:
        return pd.DataFrame(columns=["user_id", "signup_ts", "country", "device", "acq_channel"])  # type: ignore
    # sample attributes
    country = RNG.choice(COUNTRIES, size=len(signup_dates), p=[0.5, 0.1, 0.1, 0.2, 0.1])
    device = RNG.choice(DEVICES, size=len(signup_dates), p=[0.45, 0.35, 0.20])
    channel = RNG.choice(CHANNELS, size=len(signup_dates), p=[0.4, 0.45, 0.1, 0.05])
    user_id = np.array([str(uuid.uuid4()) for _ in range(len(signup_dates))])
    users = pd.DataFrame(
        {
            "user_id": user_id,
            "signup_ts": pd.to_datetime(signup_dates, utc=True),
            "country": country,
            "device": device,
            "acq_channel": channel,
        }
    )
    if len(users) > cfg.n_users:
        users = users.sample(cfg.n_users, random_state=42).reset_index(drop=True)
    return users


def heterogeneity_score(users: pd.DataFrame) -> np.ndarray:
    score = np.ones(len(users))
    # country/device/channel effects
    score *= users["country"].map({"US": 1.0, "GB": 0.95, "CA": 0.9, "DE": 0.85, "MX": 0.8}).values
    score *= users["device"].map({"ios": 1.05, "android": 1.0, "web": 0.85}).values
    score *= users["acq_channel"].map({"organic": 1.1, "referral": 1.05, "affiliate": 0.95, "paid": 0.9}).values
    # user-level random effect
    score *= RNG.lognormal(mean=0.0, sigma=0.5, size=len(users))
    return score


def simulate_events_orders(users: pd.DataFrame, cfg: SimConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    score = heterogeneity_score(users)
    max_date = cfg.start_date + timedelta(days=cfg.months * 30)
    exp_start = cfg.start_date + timedelta(days=cfg.experiment_start_offset_days)
    exp_end = exp_start + timedelta(days=cfg.experiment_days)

    events_list = []
    orders_list = []
    session_id_counter = 0

    # base rates
    base_sessions_per_week = 1.2 * score  # heterogeneous
    base_purchase_prob = 0.08 * np.clip(score / score.mean(), 0.2, 3.0)

    for idx, row in users.iterrows():
        signup: datetime = row["signup_ts"]
        user_id: str = row["user_id"]
        # tenure
        end = max_date
        weeks = max(1, math.ceil((end - signup).days / 7))
        # random decay of activity (churn-like)
        decay = RNG.uniform(0.88, 0.98)
        s_rate = base_sessions_per_week[idx]
        p_purchase = min(0.5, base_purchase_prob[idx])

        # Treatment assignment for experiment: 50% among eligible (signed up before exp_end)
        in_experiment = signup < exp_end
        treated = int(in_experiment and RNG.random() < 0.5)

        last_event_ts = None
        total_spend = 0.0
        for w in range(weeks):
            week_start = signup + timedelta(days=7 * w)
            if week_start > end:
                break
            # seasonality: higher sessions in Q4
            season = 1.0 + 0.15 * math.sin(2 * math.pi * (week_start.timetuple().tm_yday) / 365)
            lam = max(0.0, s_rate * (decay**w) * season)
            n_sessions = RNG.poisson(lam)
            for _ in range(n_sessions):
                session_id_counter += 1
                ts = week_start + timedelta(days=RNG.uniform(0, 7), hours=RNG.uniform(0, 24))
                if ts > end:
                    continue
                sid = f"s{session_id_counter}"
                events_list.append([user_id, ts, "session_start", sid, 0.0])
                # browsing events
                for _ in range(RNG.integers(1, 4)):
                    events_list.append([user_id, ts + timedelta(minutes=RNG.uniform(1, 60)), "view", sid, 0.0])
                    if RNG.random() < 0.3:
                        events_list.append([user_id, ts + timedelta(minutes=RNG.uniform(1, 60)), "add_to_cart", sid, 0.0])

                # purchase probability; uplift during experiment for persuadables
                ts_mid = ts
                uplift_pp = 0.0
                if exp_start <= ts_mid <= exp_end and in_experiment and treated:
                    # persuadable if moderate recency and mid-value (proxy from total_spend)
                    persuadable = (last_event_ts is not None and (ts_mid - last_event_ts).days > 7) and (
                        50 <= total_spend <= 300
                    )
                    if persuadable:
                        uplift_pp = RNG.uniform(0.02, 0.05)
                buy = RNG.random() < (p_purchase + uplift_pp)
                if buy:
                    amount = float(np.round(RNG.gamma(shape=2.0, scale=30.0), 2))
                    margin_rate = {"US": 0.6, "GB": 0.58, "CA": 0.55, "DE": 0.5, "MX": 0.48}[row["country"]]
                    margin = float(np.round(amount * margin_rate + RNG.normal(0, 2.0), 2))
                    order_id = str(uuid.uuid4())
                    promo_flag = int(exp_start <= ts_mid <= exp_end)
                    treatment_flag = int(treated)
                    orders_list.append([order_id, user_id, ts_mid, amount, margin, promo_flag, treatment_flag])
                    events_list.append([user_id, ts_mid, "purchase", sid, amount])
                    total_spend += amount
                last_event_ts = ts

    events = pd.DataFrame(events_list, columns=["user_id", "event_ts", "event_type", "session_id", "amount"])
    orders = pd.DataFrame(
        orders_list,
        columns=["order_id", "user_id", "order_ts", "amount", "margin", "promo_flag", "treatment_flag"],
    )
    # Deduplicate just in case
    events = events.sort_values("event_ts").reset_index(drop=True)
    orders = orders.sort_values("order_ts").reset_index(drop=True)
    return events, orders


def write_data(users: pd.DataFrame, events: pd.DataFrame, orders: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    users.to_csv(outdir / "users.csv", index=False)
    events.to_csv(outdir / "events.csv", index=False)
    orders.to_csv(outdir / "orders.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--users", type=int, default=100_000)
    ap.add_argument("--months", type=int, default=18)
    args = ap.parse_args()
    cfg = SimConfig(n_users=args.users, months=args.months)
    users = simulate_users(cfg)
    events, orders = simulate_events_orders(users, cfg)
    write_data(users, events, orders, Path(args.outdir))


if __name__ == "__main__":
    main()

