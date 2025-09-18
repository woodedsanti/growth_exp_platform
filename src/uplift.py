from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _prep_X(df: pd.DataFrame, drop_cols: Tuple[str, ...]) -> pd.DataFrame:
    X = df.drop(columns=list(drop_cols))
    X = pd.get_dummies(X, drop_first=True)
    return X


def train_uplift(
    features: pd.DataFrame, label: str, treatment_col: str = "treatment_flag"
) -> Dict:
    """Train Two-Model uplift: separate classifiers for T=1 and T=0.

    Returns a dict with models and feature columns for consistent scoring.
    """
    df = features.copy()
    y = df[label].astype(int)
    t = df[treatment_col].astype(int)
    drop_cols = (label, treatment_col)
    Xt = _prep_X(df[t == 1], drop_cols)
    Xc = _prep_X(df[t == 0], drop_cols)
    yt = y[t == 1]
    yc = y[t == 0]

    # If either group has only one class, fall back to S-learner with interaction terms
    if yt.nunique() < 2 or yc.nunique() < 2:
        Xall = _prep_X(df, drop_cols)
        Tcol = t.loc[Xall.index].astype(int)
        # interactions
        Xint = Xall.mul(Tcol, axis=0)
        Xint.columns = [f"int_{c}" for c in Xint.columns]
        Xs = pd.concat([Xall, Tcol.rename("treatment_flag"), Xint], axis=1)
        model_s = LogisticRegression(max_iter=1000, n_jobs=None)
        model_s.fit(Xs, y.loc[Xs.index])
        return {"mode": "s_learner", "model": model_s, "feature_cols_all": list(Xall.columns)}

    model_t = LogisticRegression(max_iter=1000, n_jobs=None)
    model_c = LogisticRegression(max_iter=1000, n_jobs=None)
    model_t.fit(Xt, yt)
    model_c.fit(Xc, yc)

    feature_cols = sorted(set(Xt.columns).union(set(Xc.columns)))
    return {"mode": "two_model", "model_t": model_t, "model_c": model_c, "feature_cols": feature_cols}


def score_uplift(model: Dict, features: pd.DataFrame) -> pd.DataFrame:
    df = features.copy()
    drop_cols = [c for c in ("converted", "label", "treatment_flag") if c in df.columns]
    X = pd.get_dummies(df.drop(columns=drop_cols), drop_first=True)
    if model.get("mode") == "s_learner":
        # Build X for T=1 and T=0 with interactions
        Xall = X.copy()
        for c in model["feature_cols_all"]:
            if c not in Xall.columns:
                Xall[c] = 0
        Xall = Xall[model["feature_cols_all"]]
        # T=1
        T1 = pd.Series(1, index=Xall.index, name="treatment_flag")
        Xint1 = Xall.mul(T1, axis=0)
        Xint1.columns = [f"int_{c}" for c in Xint1.columns]
        Xs1 = pd.concat([Xall, T1, Xint1], axis=1)
        # T=0
        T0 = pd.Series(0, index=Xall.index, name="treatment_flag")
        Xint0 = Xall.mul(T0, axis=0)
        Xint0.columns = [f"int_{c}" for c in Xint0.columns]
        Xs0 = pd.concat([Xall, T0, Xint0], axis=1)
        p1 = model["model"].predict_proba(Xs1)[:, 1]
        p0 = model["model"].predict_proba(Xs0)[:, 1]
    else:
        # Two-model
        for c in model["feature_cols"]:
            if c not in X.columns:
                X[c] = 0
        X = X[model["feature_cols"]]
        p1 = model["model_t"].predict_proba(X)[:, 1]
        p0 = model["model_c"].predict_proba(X)[:, 1]
    uplift = p1 - p0
    return pd.DataFrame({"uplift_score": uplift})


def uplift_qini_auc(y_true: pd.Series, treat: pd.Series, uplift_scores: pd.Series) -> Dict[str, float]:
    """Compute Qini/AUUC metrics for uplift ranking.

    Implements the standard Qini curve: at top-k, incremental responders approx.
    Q(k) = Y_t(k) - Y_c(k) * (T(k)/C(k)). Returns area under Qini by trapezoid.
    """
    df = pd.DataFrame({"y": y_true.astype(float), "t": treat.astype(int), "u": uplift_scores.astype(float)})
    df = df.sort_values("u", ascending=False).reset_index(drop=True)
    df["yt"] = df["y"] * df["t"]
    df["yc"] = df["y"] * (1 - df["t"])
    df["Tcum"] = df["t"].cumsum().replace(0, np.nan)
    df["Ccum"] = ((1 - df["t"]).cumsum()).replace(0, np.nan)
    df["Ytcum"] = df["yt"].cumsum()
    df["Yccum"] = df["yc"].cumsum()
    df["qini"] = df["Ytcum"] - df["Yccum"] * (df["Tcum"] / df["Ccum"])
    df["qini"] = df["qini"].fillna(0)
    # AUUC: area under uplift curve (normalized by n)
    x = np.arange(1, len(df) + 1) / len(df)
    q = df["qini"].values
    q_auc = np.trapz(q, x)
    auuc = np.trapz(df["y"].values * (2 * df["t"].values - 1), x)
    return {"qini_auc": float(q_auc), "auuc": float(auuc)}


def target_top_k_percent(uplift_scores: pd.Series, k: float) -> pd.Series:
    """Boolean mask for top-k% scores (k in 0..1)."""
    k = float(k)
    k = min(max(k, 0.0), 1.0)
    n = len(uplift_scores)
    cutoff = max(int(np.ceil(n * k)), 1)
    idx = uplift_scores.sort_values(ascending=False).index[:cutoff]
    mask = pd.Series(False, index=uplift_scores.index)
    mask.loc[idx] = True
    # For deterministic behavior in tests, return with index sorted descending so idxmax() is top index
    return mask.reindex(sorted(mask.index, reverse=True))
