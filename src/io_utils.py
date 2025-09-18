from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import mlflow
import pandas as pd


def get_project_root() -> Path:
    """Return project root (folder containing this file's parent)."""
    return Path(__file__).resolve().parents[1]


def _ensure_datetime(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")


def load_csv(path: str | os.PathLike, schema: Dict[str, str] | None = None) -> pd.DataFrame:
    """Load CSV with optional schema validation and basic type coercion.

    Args:
        path: CSV path.
        schema: Optional mapping col -> dtype string ("str", "float", "int", "datetime").
    """
    df = pd.read_csv(path)
    if schema:
        missing = [c for c in schema.keys() if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        for col, typ in schema.items():
            if typ == "datetime":
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            elif typ == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif typ == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            elif typ == "str":
                df[col] = df[col].astype(str)
    # Best-effort datetime inference for common columns
    _ensure_datetime(df, [c for c in df.columns if c.endswith("_ts")])
    return df


def save_csv(df: pd.DataFrame, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@dataclass
class MLflowConfig:
    experiment_name: str = "growth_exp_local"
    tracking_uri: Optional[str] = None  # default to file store under project


def _default_tracking_uri() -> str:
    mlruns_path = get_project_root() / "mlruns"
    mlruns_path.mkdir(parents=True, exist_ok=True)
    # Use proper file URI so MLflow recognizes the scheme on Windows (file:///C:/...)
    return mlruns_path.as_uri()


class MLflowHelper:
    """Minimal MLflow helper for local file-backed tracking."""

    def __init__(self, config: Optional[MLflowConfig] = None) -> None:
        self.config = config or MLflowConfig()
        self.config.tracking_uri = self.config.tracking_uri or _default_tracking_uri()
        mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

    def start_run(self, run_name: str | None = None):
        return mlflow.start_run(run_name=run_name)

    @staticmethod
    def log_dict_artifact(d: Dict[str, Any], artifact_path: str) -> None:
        tmp = get_project_root() / "_tmp_artifact.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        mlflow.log_artifact(str(tmp), artifact_path=artifact_path)
        tmp.unlink(missing_ok=True)

    @staticmethod
    def log_params(d: Dict[str, Any]) -> None:
        mlflow.log_params({k: (str(v) if isinstance(v, (dict, list)) else v) for k, v in d.items()})

    @staticmethod
    def log_metrics(d: Dict[str, float]) -> None:
        mlflow.log_metrics(d)
