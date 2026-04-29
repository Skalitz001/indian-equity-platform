import pandas as pd

from src.validation.prediction_store import (
    PREDICTION_COLUMNS,
    build_prediction_frame,
    save_prediction_frame,
)


def test_build_prediction_frame_uses_prediction_schema_and_down_probability(tmp_path):
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2, freq="B"),
            "target": [1, 0],
            "target_return": [0.02, -0.01],
        },
        index=[10, 11],
    )
    probabilities = pd.Series([0.70, 0.35], index=[10, 11])
    signals = pd.Series([1, 0], index=[10, 11])

    out = build_prediction_frame(
        df,
        probabilities,
        signals,
        ticker="ABC.NS",
        pipeline="main",
        model_name="logistic",
        model_label="logreg",
        feature_group="baseline",
        split="holdout",
        entry_threshold=0.60,
        exit_threshold=0.55,
        return_column="target_return",
    )

    assert out.columns.tolist() == PREDICTION_COLUMNS
    assert out["probability_down"].round(2).tolist() == [0.30, 0.65]
    assert out["realized_direction"].tolist() == ["UP", "DOWN"]
    assert out["realized_return"].tolist() == [0.02, -0.01]

    path = save_prediction_frame(
        out,
        artifacts_dir=tmp_path,
        ticker="ABC.NS",
        pipeline="main",
        model_name="logistic",
    )

    saved = pd.read_csv(path)
    assert saved.columns.tolist() == PREDICTION_COLUMNS
    assert path.name == "ABC_NS_main_logistic_predictions.csv"
