import numpy as np
import pandas as pd

from src.analytics.changepoint_features import (
    CHANGEPOINT_FEATURE_COLUMNS,
    PAPER_CHANGEPOINT_FEATURE_COLUMNS,
    build_changepoint_feature_frame,
)


def make_price_frame(periods: int = 150) -> pd.DataFrame:
    base = np.linspace(100, 135, periods)

    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=periods, freq="B"),
        "Open": base * 0.997,
        "High": base * 1.012,
        "Low": base * 0.988,
        "Close": base,
        "Volume": np.linspace(1_000_000, 1_400_000, periods),
    })


def test_build_changepoint_feature_frame_contains_features():
    df = make_price_frame()

    out = build_changepoint_feature_frame(df)

    assert set(CHANGEPOINT_FEATURE_COLUMNS).issubset(out.columns)
    assert set(PAPER_CHANGEPOINT_FEATURE_COLUMNS).issubset(CHANGEPOINT_FEATURE_COLUMNS)
    assert out[CHANGEPOINT_FEATURE_COLUMNS].isnull().sum().sum() == 0


def test_changepoint_features_do_not_use_future_rows():
    df = make_price_frame()
    changed = df.copy()
    changed.loc[120:, "Close"] = changed.loc[120:, "Close"] * 1.25

    original_out = build_changepoint_feature_frame(df, dropna=False)
    changed_out = build_changepoint_feature_frame(changed, dropna=False)

    pd.testing.assert_frame_equal(
        original_out.loc[:110, CHANGEPOINT_FEATURE_COLUMNS],
        changed_out.loc[:110, CHANGEPOINT_FEATURE_COLUMNS],
    )
