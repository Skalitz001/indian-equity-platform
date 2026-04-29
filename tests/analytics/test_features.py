import numpy as np
import pandas as pd
from src.analytics.features import (
    FEATURE_COLUMNS,
    add_log_returns,
    add_sma,
    build_feature_frame,
)


def test_add_log_returns():
    df = pd.DataFrame({"Close": [100, 101, 102]})
    out = add_log_returns(df)

    assert "log_return" in out.columns
    assert out["log_return"].isna().sum() == 1


def test_add_sma():
    df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]})
    out = add_sma(df, window=3)

    assert "sma_3" in out.columns
    assert out["sma_3"].iloc[2] == 2.0


def test_build_feature_frame_contains_model_features():
    periods = 90
    base = np.linspace(100, 130, periods)

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=periods, freq="B"),
            "Open": base * 0.995,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base,
            "Volume": np.linspace(1_000_000, 1_250_000, periods),
        }
    )

    out = build_feature_frame(df)

    assert set(FEATURE_COLUMNS).issubset(out.columns)
    assert out[FEATURE_COLUMNS].isnull().sum().sum() == 0
