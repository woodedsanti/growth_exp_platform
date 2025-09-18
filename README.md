# Growth Analytics & Experimentation Platform

Enable a product/growth team to:

- See retention health by cohort/segment
- Forecast 6/12-month LTV
- Plan & evaluate experiments (power/MDE, CUPED)
- Target promotions to persuadable users via uplift modeling to maximize net revenue lift

## Repo Layout

See `growth_exp_platform/` for code, data, app, tests, and docs. Core modules live in `src/` and are reused across Streamlit and notebooks.

## Tech Stack

- Python 3.11
- pandas, numpy, scikit-learn, statsmodels, lifetimes, scipy
- Plotly + Streamlit for UI
- MLflow for run tracking (local file backend)
- pytest for tests
- pre-commit (black, isort, flake8) for linting

## Data Dictionary

All timestamps are ISO8601 UTC.

- `users.csv`: `user_id`, `signup_ts`, `country` (US, MX, CA, GB, DE), `device` (ios, android, web), `acq_channel` (paid, organic, referral, affiliate)
- `events.csv`: `user_id`, `event_ts`, `event_type` (session_start, view, add_to_cart, purchase), `session_id`, `amount` (0 if not purchase)
- `orders.csv`: `order_id`, `user_id`, `order_ts`, `amount`, `margin`, `promo_flag`, `treatment_flag`

## How To Run

1) Create/activate a Python 3.11 env.

2) From `growth_exp_platform/` run:

- `make setup` – install deps and pre-commit
- `make data` – generate synthetic CSVs into `data/` (100k users)
- `make test` – run pytest
- `make app` – launch Streamlit app at `http://localhost:8501`
- `make mlflow` – start local MLflow UI at `http://127.0.0.1:5000`

Notes: On Windows, install `make` or run the underlying commands manually. No external data calls are made; all data is generated locally.

## Methods (Notes)

- Cohorts & retention: user is “retained” in week k if they have ≥1 session in that week window since signup. We build a weekly cohort matrix and render a Plotly heatmap.
- LTV: BG/NBD (repeat transactions) + Gamma-Gamma (monetary value). We fit on calibration and evaluate on a 60-day holdout (MAPE/WMAPE). Alternative survival/GLM path is included for illustration.
- Experiments: Power/MDE calculators for proportions and means (statsmodels). CUPED variance reduction with Y_adj = Y - theta (X - X̄).
- Uplift: Two-Model approach with separate classifiers for treatment and control; uplift = p1 - p0. We compute the Qini curve and AUUC and provide a targeting simulator for net revenue lift.

## App Screens

Screenshots can be saved from Streamlit into `reports/screenshots/` as needed.

## Limitations & Next Steps

- Uplift modeling can be upgraded with doubly robust learners / causal forests.
- Hierarchical LTV by segment with partial pooling.
- CUPED with multiple covariates (OLS/GLM) and heteroscedasticity-robust SEs.
- Deeper retention diagnostics (survival curves by cohort, seasonality).

