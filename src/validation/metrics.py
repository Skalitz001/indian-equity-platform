import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


CLASSIFICATION_THRESHOLD_GRID = np.linspace(0.45, 0.65, 21)


def _as_series(values, name: str, index=None) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.rename(name)

    return pd.Series(values, index=index, name=name)


def _aligned_frame(y_true, values, value_name: str) -> pd.DataFrame:
    y_true_series = _as_series(y_true, "y_true")
    value_index = (
        y_true_series.index
        if not isinstance(values, pd.Series) and len(values) == len(y_true_series)
        else None
    )
    value_series = _as_series(values, value_name, index=value_index)

    return pd.concat(
        [y_true_series, value_series],
        axis=1,
    ).dropna()


def classification_metrics(
    y_true: pd.Series, y_pred: pd.Series
) -> dict:
    frame = _aligned_frame(y_true, y_pred, "y_pred")

    if frame.empty:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    y_true_values = frame["y_true"].astype(int)
    y_pred_values = frame["y_pred"].astype(int)

    return {
        "accuracy": accuracy_score(y_true_values, y_pred_values),
        "precision": precision_score(
            y_true_values,
            y_pred_values,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true_values,
            y_pred_values,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true_values,
            y_pred_values,
            zero_division=0,
        ),
    }


def threshold_predictions(probas, threshold: float) -> pd.Series:
    probabilities = _as_series(probas, "probability_up").astype(float)
    return (probabilities > float(threshold)).astype(int)


def find_best_threshold(y_true, probas, thresholds=None):
    frame = _aligned_frame(y_true, probas, "probability_up")

    if frame.empty:
        return 0.5

    thresholds = (
        CLASSIFICATION_THRESHOLD_GRID
        if thresholds is None else np.asarray(thresholds, dtype=float)
    )

    best_t, best_f1 = 0.5, -1.0
    y_true_values = frame["y_true"].astype(int)
    probabilities = frame["probability_up"].astype(float)

    for t in thresholds:
        preds = threshold_predictions(probabilities, t)
        f1 = f1_score(y_true_values, preds, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t


def classification_metrics_from_probabilities(
    y_true,
    probas,
    threshold: float | None = None,
    optimize_threshold: bool = False,
    thresholds=None,
) -> dict:
    frame = _aligned_frame(y_true, probas, "probability_up")

    if frame.empty:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "threshold": float(0.5 if threshold is None else threshold),
            "positive_rate": 0.0,
            "support": 0,
        }

    threshold_value = (
        find_best_threshold(
            frame["y_true"],
            frame["probability_up"],
            thresholds=thresholds,
        )
        if optimize_threshold
        else float(0.5 if threshold is None else threshold)
    )

    predictions = threshold_predictions(
        frame["probability_up"],
        threshold_value,
    )
    metrics = classification_metrics(frame["y_true"], predictions)

    metrics.update({
        "threshold": float(threshold_value),
        "positive_rate": float(predictions.mean()),
        "support": int(len(predictions)),
    })

    return metrics
