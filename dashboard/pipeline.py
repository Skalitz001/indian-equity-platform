import pandas as pd
import numpy as np
from joblib import load

from dashboard.config import (
    GIFT_ARTIFACTS_DIR,
    GIFT_COMPARISON_SUMMARY_PATH,
    MAIN_ARTIFACTS_DIR,
)
from src.analytics.experimental_features import build_main_feature_frame
from src.validation.metrics import classification_metrics_from_probabilities


def build_feature_frame(df, feature_group="baseline"):
    return build_main_feature_frame(
        df,
        feature_group=feature_group,
    )


def add_intraday_returns(df):
    frame = df.copy()
    open_price = frame["Open"].replace(0, np.nan)
    frame["intraday_return"] = (frame["Close"] / open_price) - 1

    return frame


def dashboard_tickers(repo, pipeline_key):
    if pipeline_key == "gift":
        return sorted(
            path.name.replace("_gift_meta.json", "").replace("_", ".")
            for path in GIFT_ARTIFACTS_DIR.glob("*_gift_meta.json")
        )

    return repo.list_tickers()


def artifacts_dir_for_pipeline(pipeline_key):
    if pipeline_key == "gift":
        return GIFT_ARTIFACTS_DIR

    return MAIN_ARTIFACTS_DIR


def meta_path_for_pipeline(ticker, pipeline_key):
    suffix = "_gift_meta.json" if pipeline_key == "gift" else "_meta.json"

    return (
        artifacts_dir_for_pipeline(pipeline_key)
        / f"{ticker.replace('.', '_')}{suffix}"
    )


def load_prediction_model(meta, model_name, artifacts_dir):
    log_path = artifacts_dir / meta["log_model"]
    rf_path = artifacts_dir / meta["rf_model"]

    if model_name == "logistic":
        return load(log_path)

    if model_name == "rf":
        return load(rf_path)

    if model_name == "ensemble":
        return {
            "kind": "ensemble",
            "log_model": load(log_path),
            "rf_model": load(rf_path),
            "weights": meta.get(
                "blend_weights",
                {
                    "logistic": 0.5,
                    "rf": 0.5,
                },
            ),
        }

    raise ValueError(f"Unsupported model '{model_name}'")


def resolve_classifier_metrics(meta, model_key, analysis_df):
    stored_metrics = meta.get(
        "oof_classification_metrics",
        {},
    ).get(model_key)

    required_fields = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "threshold",
        "positive_rate",
        "support",
    }

    if stored_metrics and required_fields.issubset(stored_metrics):
        metrics = {
            "accuracy": float(stored_metrics["accuracy"]),
            "precision": float(stored_metrics["precision"]),
            "recall": float(stored_metrics["recall"]),
            "f1": float(stored_metrics["f1"]),
            "threshold": float(stored_metrics["threshold"]),
            "positive_rate": float(stored_metrics["positive_rate"]),
            "support": int(stored_metrics["support"]),
        }

        note = (
            "Out-of-fold classifier metrics from walk-forward training. "
            "These do not use the Sharpe-optimized trade state machine."
        )

        return metrics, note

    metrics = classification_metrics_from_probabilities(
        analysis_df["target"],
        analysis_df["probability_up"],
        threshold=0.50,
    )

    note = (
        "Fallback classifier metrics for the selected period using a fixed "
        "0.50 probability threshold because OOF classifier metadata is unavailable."
    )

    return metrics, note


def load_comparison_summary():
    if not GIFT_COMPARISON_SUMMARY_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(GIFT_COMPARISON_SUMMARY_PATH)
