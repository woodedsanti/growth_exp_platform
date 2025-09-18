from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ab_tools import required_sample_size_means, required_sample_size_proportions
from src.cohorts import make_cohorts, plot_retention_heatmap
from src.cuped import cuped_adjust, cuped_effect_summary
from src.features import build_user_features, make_experiment_dataset
from src.io_utils import MLflowHelper, get_project_root, load_csv
from src.ltv import fit_bgnbd_gg, prepare_transactions, predict_ltv
from src.metrics import diff_in_means, diff_in_proportions, mape, wmape
from src.uplift import score_uplift, target_top_k_percent, train_uplift, uplift_qini_auc


st.set_page_config(page_title="Growth Analytics & Experimentation", layout="wide")


@st.cache_data
def load_sample_data():
    root = get_project_root()
    data_dir = root / "data"
    users = load_csv(data_dir / "users.csv", {
        "user_id": "str", "signup_ts": "datetime", "country": "str", "device": "str", "acq_channel": "str"
    })
    events = load_csv(data_dir / "events.csv", {
        "user_id": "str", "event_ts": "datetime", "event_type": "str", "session_id": "str", "amount": "float"
    })
    orders = load_csv(data_dir / "orders.csv", {
        "order_id": "str", "user_id": "str", "order_ts": "datetime", "amount": "float",
        "margin": "float", "promo_flag": "int", "treatment_flag": "int"
    })
    return users, events, orders


def sidebar_docs():
    st.sidebar.header("Data Dictionary")
    st.sidebar.markdown("""
    - users.csv: user_id, signup_ts, country, device, acq_channel
    - events.csv: user_id, event_ts, event_type, session_id, amount
    - orders.csv: order_id, user_id, order_ts, amount, margin, promo_flag, treatment_flag
    """)
    st.sidebar.header("Assumptions")
    st.sidebar.info(
        "Weekly retention: active if ≥1 session in period. BG/NBD assumes stationarity; CUPED uses a single covariate; Uplift via Two-Model."
    )


def tab_cohorts():
    st.header("Cohorts")
    up_users = st.file_uploader("Upload users.csv", type="csv")
    up_events = st.file_uploader("Upload events.csv", type="csv")
    if up_users and up_events:
        users = pd.read_csv(up_users)
        events = pd.read_csv(up_events)
    else:
        users, events, _ = load_sample_data()

    freq = st.selectbox("Frequency", options=["W", "M"], index=0)
    seg_col = st.selectbox("Segment by", options=["country", "device", "acq_channel", None], index=0)
    if seg_col:
        for val in sorted(users[seg_col].unique()):
            with st.expander(f"Segment: {seg_col} = {val}"):
                u = users[users[seg_col] == val]
                cohort_df = make_cohorts(events, u, freq=freq)
                st.plotly_chart(plot_retention_heatmap(cohort_df), use_container_width=True)
                # KPIs W4/W8/W12
                mat = cohort_df.reset_index().pivot_table(index="cohort", columns="period", values="retention_rate")
                w4 = mat.get(4, pd.Series(dtype=float)).mean()
                w8 = mat.get(8, pd.Series(dtype=float)).mean()
                w12 = mat.get(12, pd.Series(dtype=float)).mean()
                st.write({"week4": round(float(w4 or 0), 4), "week8": round(float(w8 or 0), 4), "week12": round(float(w12 or 0), 4)})
                st.dataframe(cohort_df.reset_index().groupby("cohort")["n_users"].max())
    else:
        cohort_df = make_cohorts(events, users, freq=freq)
        st.plotly_chart(plot_retention_heatmap(cohort_df), use_container_width=True)
        mat = cohort_df.reset_index().pivot_table(index="cohort", columns="period", values="retention_rate")
        w4 = mat.get(4, pd.Series(dtype=float)).mean()
        w8 = mat.get(8, pd.Series(dtype=float)).mean()
        st.write({"week4": round(float(w4 or 0), 4), "week8": round(float(w8 or 0), 4)})
        st.dataframe(cohort_df.reset_index().groupby("cohort")["n_users"].max())


def tab_ltv():
    st.header("LTV")
    users, events, orders = load_sample_data()
    horizon = st.selectbox("Horizon (days)", options=[180, 365], index=0)
    # Holdout last 60 days
    cutoff = pd.to_datetime(orders["order_ts"]).max() - pd.Timedelta(days=60)
    cal = orders[pd.to_datetime(orders["order_ts"]) <= cutoff]
    holdout = orders[pd.to_datetime(orders["order_ts"]) > cutoff]

    tx = prepare_transactions(cal)
    models = fit_bgnbd_gg(tx)
    preds = predict_ltv(models, horizon_days=int(horizon), user_df=tx.merge(cal[["user_id"]].drop_duplicates(), on="user_id"))

    # Aggregate actuals in holdout per user
    actual = holdout.groupby("user_id")["amount"].sum().rename("actual").reset_index()
    pred_col = [c for c in preds.columns if c.startswith("ltv_h")][0]
    eval_df = preds.merge(actual, on="user_id", how="left").fillna({"actual": 0.0})
    kpis = {"MAPE": mape(eval_df["actual"], eval_df[pred_col]), "WMAPE": wmape(eval_df["actual"], eval_df[pred_col])}
    st.write({k: round(float(v), 4) for k, v in kpis.items()})

    fig = px.scatter(eval_df, x=pred_col, y="actual", title="Predicted vs Actual (Holdout)", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

    # Log to MLflow
    mlh = MLflowHelper()
    with mlh.start_run(run_name="ltv_bg-nbd"):
        mlh.log_params({"horizon_days": int(horizon)})
        mlh.log_metrics({"mape": float(kpis["MAPE"]), "wmape": float(kpis["WMAPE"])})


def tab_experiments():
    st.header("Experiments")
    st.subheader("Power/MDE Calculators")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Proportions (conversion rate)")
        p = st.number_input("Baseline rate", value=0.05, min_value=0.0001, max_value=0.9999, step=0.005)
        mde = st.number_input("MDE (abs)", value=0.01, min_value=0.0001, max_value=0.9999, step=0.005)
        n = required_sample_size_proportions(p, mde)
        st.write({"required_per_arm": int(n)})
    with c2:
        st.markdown("Means (e.g., revenue)")
        sd = st.number_input("SD", value=1.0, min_value=1e-6, step=0.1)
        mde_m = st.number_input("MDE (abs)", value=0.2, min_value=1e-6, step=0.05)
        n2 = required_sample_size_means(sd, mde_m)
        st.write({"required_per_arm": int(n2)})

    st.subheader("CUPED Variance Reduction")
    up = st.file_uploader("Upload experiment CSV (user_id,treatment_flag,pre_x,outcome_y)", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        s = cuped_effect_summary(df["outcome_y"], df["pre_x"], df["treatment_flag"])
        st.write(s)
    else:
        # demo synthetic
        rng = np.random.default_rng(0)
        n = 5000
        t = rng.integers(0, 2, size=n)
        x = rng.normal(0, 1, size=n)
        y0 = 0.2 * x + rng.normal(0, 1, size=n)
        y = y0 + 0.1 * t
        df = pd.DataFrame({"treatment_flag": t, "pre_x": x, "outcome_y": y})
        s = cuped_effect_summary(df["outcome_y"], df["pre_x"], df["treatment_flag"])
        st.write(s)


def tab_targeting():
    st.header("Targeting (Uplift)")
    users, events, orders = load_sample_data()
    # detect experiment window from data
    if len(orders) == 0:
        st.warning("No orders found in sample data.")
        return
    exp_orders = orders[orders["promo_flag"] == 1]
    if len(exp_orders) == 0:
        exp_orders = orders.tail(1000)
    ws = pd.to_datetime(exp_orders["order_ts"]).min()
    we = ws + pd.Timedelta(days=28)
    data = make_experiment_dataset(users, events, orders, ws, we)
    if st.button("Train uplift model"):
        # Proceed; train_uplift will fall back to S-learner with interactions if needed
        model = train_uplift(data.rename(columns={"converted": "label"}), label="label", treatment_col="treatment_flag")
        if model.get("mode") == "s_learner":
            st.info("Fell back to S-learner with interactions due to low class variety in a group.")
        scores = score_uplift(model, data)
        met = uplift_qini_auc(data["converted"], data["treatment_flag"], scores["uplift_score"])
        st.write({k: round(float(v), 4) for k, v in met.items()})
        # Log to MLflow
        mlh = MLflowHelper()
        with mlh.start_run(run_name="uplift_two_model"):
            mlh.log_metrics({"qini_auc": float(met["qini_auc"]), "auuc": float(met["auuc"])})
        # Uplift curve
        df = data.copy()
        df["uplift"] = scores["uplift_score"].values
        df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
        df["cum_t"] = df["treatment_flag"].cumsum().replace(0, np.nan)
        df["cum_c"] = ((1 - df["treatment_flag"]).cumsum()).replace(0, np.nan)
        df["cum_y_t"] = (df["converted"] * df["treatment_flag"]).cumsum()
        df["cum_y_c"] = (df["converted"] * (1 - df["treatment_flag"]))
        df["cum_y_c"] = df["cum_y_c"].cumsum()
        df["qini"] = df["cum_y_t"] - df["cum_y_c"] * (df["cum_t"] / df["cum_c"])
        fig = px.line(df.reset_index(), x=df.index / len(df), y="qini", labels={"x": "share targeted", "y": "Qini"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Net Revenue Lift Simulator")
        discount = st.number_input("Discount %", value=10.0, min_value=0.0, max_value=100.0, step=1.0)
        unit_margin = st.number_input("Unit margin ($)", value=20.0, min_value=0.0, step=1.0)
        audience = st.slider("Max audience %", 1, 100, 20)
        mask = target_top_k_percent(df["uplift"], k=audience / 100.0)
        inc_conversions = float(df.loc[mask, "uplift"].sum())
        promo_cost = float(mask.sum() * (discount / 100.0) * unit_margin)
        net_lift = inc_conversions * unit_margin - promo_cost
        st.write({"incremental_conversions": round(inc_conversions, 2), "promo_cost": round(promo_cost, 2), "net_lift": round(net_lift, 2)})

        # Export CSV
        top = df.loc[mask, ["user_id", "uplift"]].rename(columns={"uplift": "uplift_score"})
        csv = top.to_csv(index=False).encode("utf-8")
        st.download_button("Export target list (CSV)", data=csv, file_name="target_list.csv", mime="text/csv")


def main():
    sidebar_docs()
    tabs = st.tabs(["Cohorts", "LTV", "Experiments", "Targeting"])
    with tabs[0]:
        tab_cohorts()
    with tabs[1]:
        tab_ltv()
    with tabs[2]:
        tab_experiments()
    with tabs[3]:
        tab_targeting()


if __name__ == "__main__":
    main()
