import pandas as pd

from src.validation.metrics import (
    classification_metrics_from_probabilities,
    find_best_threshold,
)


def test_find_best_threshold_uses_f1_on_aligned_probabilities():
    y_true = pd.Series([1, 1, 0, 0], index=[10, 11, 12, 13])
    probabilities = pd.Series(
        [0.70, 0.46, 0.45, 0.44],
        index=[10, 11, 12, 13],
    )

    threshold = find_best_threshold(
        y_true,
        probabilities,
        thresholds=[0.45, 0.50],
    )

    assert threshold == 0.45


def test_classification_metrics_from_probabilities_preserves_index_alignment():
    y_true = pd.Series([1, 0, 1], index=[2, 4, 6])

    metrics = classification_metrics_from_probabilities(
        y_true,
        [0.9, 0.4, 0.3],
        threshold=0.50,
    )

    assert metrics == {
        "accuracy": 2 / 3,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "threshold": 0.5,
        "positive_rate": 1 / 3,
        "support": 3,
    }
