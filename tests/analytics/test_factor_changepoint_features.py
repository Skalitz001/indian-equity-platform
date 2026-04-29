import numpy as np
import pandas as pd

from src.analytics.changepoint_features import add_changepoint_features
from src.analytics.experimental_features import build_experimental_feature_frame
from src.analytics.factor_changepoint_features import (
    FACTOR_CHANGEPOINT_FEATURE_COLUMNS,
    add_factor_residual_changepoint_features,
    build_cross_sectional_factor_frame,
)


def make_price_frame(
    periods: int = 180,
    offset: float = 0.0,
) -> pd.DataFrame:
    base = np.linspace(100 + offset, 145 + offset, periods)
    seasonal = 1.0 + (0.01 * np.sin(np.arange(periods) / 5.0))
    close = base * seasonal

    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=periods, freq="B"),
        "Open": close * 0.997,
        "High": close * 1.012,
        "Low": close * 0.988,
        "Close": close,
        "Volume": np.linspace(1_000_000, 1_400_000, periods),
    })


def build_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = build_experimental_feature_frame(df, dropna=False)
    return add_changepoint_features(frame)


def test_factor_changepoint_features_do_not_use_future_peer_rows():
    feature_frames = {
        "AAA.NS": build_frame(make_price_frame(offset=0.0)),
        "BBB.NS": build_frame(make_price_frame(offset=10.0)),
        "CCC.NS": build_frame(make_price_frame(offset=20.0)),
    }
    changed_feature_frames = {
        ticker: frame.copy()
        for ticker, frame in feature_frames.items()
    }
    changed_feature_frames["BBB.NS"].loc[130:, "log_return"] *= 3.0
    changed_feature_frames["CCC.NS"].loc[130:, "log_return"] *= -2.0

    factor = build_cross_sectional_factor_frame(
        feature_frames,
        exclude_ticker="AAA.NS",
    )
    changed_factor = build_cross_sectional_factor_frame(
        changed_feature_frames,
        exclude_ticker="AAA.NS",
    )

    out = add_factor_residual_changepoint_features(
        feature_frames["AAA.NS"],
        factor,
    )
    changed_out = add_factor_residual_changepoint_features(
        changed_feature_frames["AAA.NS"],
        changed_factor,
    )

    pd.testing.assert_frame_equal(
        out.loc[:110, FACTOR_CHANGEPOINT_FEATURE_COLUMNS],
        changed_out.loc[:110, FACTOR_CHANGEPOINT_FEATURE_COLUMNS],
    )
