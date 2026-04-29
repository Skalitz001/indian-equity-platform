from pathlib import Path

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = [
    "Date",
    "ticker",
    "pipeline",
    "model_name",
    "model_label",
    "feature_group",
    "split",
    "probability_up",
    "probability_down",
    "entry_threshold",
    "exit_threshold",
    "signal",
    "target",
    "realized_direction",
    "realized_return",
]


def build_prediction_frame(
    df: pd.DataFrame,
    probabilities,
    signals,
    *,
    ticker: str,
    pipeline: str,
    model_name: str,
    model_label: str,
    feature_group: str,
    split: str,
    entry_threshold: float,
    exit_threshold: float,
    return_column: str | None = None,
) -> pd.DataFrame:
    probability_series = pd.Series(probabilities, dtype=float)
    signal_series = pd.Series(signals, index=probability_series.index).astype(int)
    aligned = df.reindex(probability_series.index).copy()

    output = pd.DataFrame(index=probability_series.index)
    output["Date"] = pd.to_datetime(aligned["Date"]).dt.strftime("%Y-%m-%d")
    output["ticker"] = ticker
    output["pipeline"] = pipeline
    output["model_name"] = model_name
    output["model_label"] = model_label
    output["feature_group"] = feature_group
    output["split"] = split
    output["probability_up"] = probability_series.astype(float)
    output["probability_down"] = 1.0 - output["probability_up"]
    output["entry_threshold"] = float(entry_threshold)
    output["exit_threshold"] = float(exit_threshold)
    output["signal"] = signal_series
    output["target"] = aligned["target"].astype(int)
    output["realized_direction"] = np.where(output["target"] == 1, "UP", "DOWN")

    if return_column and return_column in aligned.columns:
        output["realized_return"] = aligned[return_column].astype(float)
    elif "target_return" in aligned.columns:
        output["realized_return"] = aligned["target_return"].astype(float)
    else:
        output["realized_return"] = np.nan

    return output[PREDICTION_COLUMNS].reset_index(drop=True)


def save_prediction_frame(
    frame: pd.DataFrame,
    *,
    artifacts_dir: Path,
    ticker: str,
    pipeline: str,
    model_name: str,
) -> Path:
    output_dir = artifacts_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_ticker = ticker.replace(".", "_").replace("^", "INDEX_")
    path = output_dir / f"{safe_ticker}_{pipeline}_{model_name}_predictions.csv"
    frame[PREDICTION_COLUMNS].to_csv(path, index=False)
    return path
