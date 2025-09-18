import pandas as pd

from src.cohorts import make_cohorts


def test_make_cohorts_weekly():
    # Toy users: 2 users in same week cohort
    users = pd.DataFrame(
        {
            "user_id": ["u1", "u2"],
            "signup_ts": ["2023-01-01T00:00:00Z", "2023-01-02T00:00:00Z"],
            "country": ["US", "US"],
            "device": ["ios", "android"],
            "acq_channel": ["organic", "paid"],
        }
    )
    events = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "event_ts": [
                "2023-01-03T00:00:00Z",  # week 0
                "2023-01-10T00:00:00Z",  # week 1
                "2023-01-09T00:00:00Z",  # week 1
            ],
            "event_type": ["session_start", "session_start", "session_start"],
            "session_id": ["s1", "s2", "s3"],
            "amount": [0.0, 0.0, 0.0],
        }
    )
    df = make_cohorts(events, users, freq="W").reset_index()
    # Both in same cohort week
    cohort_key = pd.to_datetime("2022-12-26T00:00:00Z")  # week starting Monday
    w0 = df[(df["cohort"] == cohort_key) & (df["period"] == 0)].iloc[0]
    w1 = df[(df["cohort"] == cohort_key) & (df["period"] == 1)].iloc[0]
    assert w0["n_users"] == 2
    # week0: only u1 retained
    assert w0["retained_users"] == 1
    # week1: both retained
    assert w1["retained_users"] == 2

