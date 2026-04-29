import numpy as np
import pandas as pd

from src.analytics.changepoint_features import PAPER_CHANGEPOINT_FEATURE_COLUMNS
from src.analytics.experimental_features import (
    DEFAULT_TRADING_FEATURE_GROUP,
    MARKET_CONTEXT_FEATURE_COLUMNS,
    resolve_main_feature_columns,
)
from src.gift_nifty.dataset import (
    GIFT_MODEL_FEATURE_COLUMNS,
    GIFT_PRE_OPEN_SOURCE_ASSUMPTION,
    GIFT_STOCK_FEATURE_COLUMNS,
    GIFT_STOCK_FEATURE_GROUP,
    build_gift_model_frame,
)
from src.gift_nifty.repository import normalize_gift_history


def test_normalize_gift_history_accepts_common_public_column_names():
    raw = pd.DataFrame(
        {
            "Date": ["2023-07-03", "2023-07-04"],
            "Price": ["19,400.50", "19,525.00"],
            "Open": ["19,350.00", "19,410.00"],
            "High": ["19,450.00", "19,560.00"],
            "Low": ["19,300.00", "19,390.00"],
            "Vol.": ["1.2K", "-"],
        }
    )

    out = normalize_gift_history(raw)

    assert out.columns.tolist() == [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    assert out["Close"].tolist() == [19400.5, 19525.0]
    assert out["Volume"].tolist() == [1200.0, 0.0]


def test_build_gift_model_frame_uses_previous_stock_features():
    periods = 120
    stock_base = np.linspace(100, 140, periods)
    gift_base = np.linspace(19000, 20100, periods)
    dates = pd.date_range("2023-07-03", periods=periods, freq="B")

    stock_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": stock_base * 0.998,
            "High": stock_base * 1.01,
            "Low": stock_base * 0.99,
            "Close": stock_base,
            "Volume": np.linspace(1_000_000, 1_250_000, periods),
        }
    )

    gift_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": gift_base * 0.999,
            "High": gift_base * 1.005,
            "Low": gift_base * 0.995,
            "Close": gift_base,
            "Volume": np.zeros(periods),
        }
    )

    out = build_gift_model_frame(stock_df, gift_df)

    assert set(GIFT_MODEL_FEATURE_COLUMNS).issubset(out.columns)
    assert out[GIFT_MODEL_FEATURE_COLUMNS].isnull().sum().sum() == 0
    assert (out["gift_source_date"] <= out["Date"]).all()
    assert (out["gift_source_age_days"] >= 0).all()


def test_gift_stock_features_match_main_trading_feature_group():
    assert GIFT_STOCK_FEATURE_GROUP == DEFAULT_TRADING_FEATURE_GROUP
    assert GIFT_STOCK_FEATURE_COLUMNS == resolve_main_feature_columns(
        DEFAULT_TRADING_FEATURE_GROUP,
    )

    stock_feature_columns = {
        column.replace("stock_prev_", "", 1)
        for column in GIFT_MODEL_FEATURE_COLUMNS
        if column.startswith("stock_prev_")
    }

    assert set(MARKET_CONTEXT_FEATURE_COLUMNS).issubset(stock_feature_columns)
    assert set(PAPER_CHANGEPOINT_FEATURE_COLUMNS).issubset(stock_feature_columns)


def test_gift_stock_changepoint_features_do_not_use_future_stock_rows():
    periods = 160
    stock_base = np.linspace(100, 150, periods)
    gift_base = np.linspace(19000, 20300, periods)
    dates = pd.date_range("2023-07-03", periods=periods, freq="B")

    stock_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": stock_base * 0.998,
            "High": stock_base * 1.01,
            "Low": stock_base * 0.99,
            "Close": stock_base,
            "Volume": np.linspace(1_000_000, 1_300_000, periods),
        }
    )

    gift_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": gift_base * 0.999,
            "High": gift_base * 1.005,
            "Low": gift_base * 0.995,
            "Close": gift_base,
            "Volume": np.zeros(periods),
        }
    )

    changed_stock_df = stock_df.copy()
    changed_stock_df.loc[125:, "Close"] = changed_stock_df.loc[125:, "Close"] * 1.30

    original = build_gift_model_frame(stock_df, gift_df)
    changed = build_gift_model_frame(changed_stock_df, gift_df)

    stock_prev_columns = [
        f"stock_prev_{column}"
        for column in GIFT_STOCK_FEATURE_COLUMNS
    ]
    compare_columns = ["Date", *stock_prev_columns]

    original_slice = original.loc[
        original["Date"] <= dates[110],
        compare_columns,
    ].reset_index(drop=True)
    changed_slice = changed.loc[
        changed["Date"] <= dates[110],
        compare_columns,
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(original_slice, changed_slice)


def test_gift_model_frame_uses_same_day_gift_as_pre_open_signal():
    periods = 130
    stock_base = np.linspace(100, 145, periods)
    gift_base = np.linspace(19000, 20400, periods)
    dates = pd.date_range("2023-07-03", periods=periods, freq="B")

    stock_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": stock_base * 0.998,
            "High": stock_base * 1.01,
            "Low": stock_base * 0.99,
            "Close": stock_base,
            "Volume": np.linspace(1_000_000, 1_250_000, periods),
        }
    )
    gift_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": gift_base * 0.999,
            "High": gift_base * 1.005,
            "Low": gift_base * 0.995,
            "Close": gift_base,
            "Volume": np.zeros(periods),
        }
    )

    out = build_gift_model_frame(stock_df, gift_df)
    target_date = dates[100]
    row = out.loc[out["Date"] == target_date].iloc[0]

    assert "pre-open" in GIFT_PRE_OPEN_SOURCE_ASSUMPTION
    assert row["gift_source_date"] == target_date
    assert row["gift_source_age_days"] == 0.0


def test_gift_features_do_not_use_future_gift_rows():
    periods = 160
    stock_base = np.linspace(100, 150, periods)
    gift_base = np.linspace(19000, 20300, periods)
    dates = pd.date_range("2023-07-03", periods=periods, freq="B")

    stock_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": stock_base * 0.998,
            "High": stock_base * 1.01,
            "Low": stock_base * 0.99,
            "Close": stock_base,
            "Volume": np.linspace(1_000_000, 1_300_000, periods),
        }
    )
    gift_df = pd.DataFrame(
        {
            "Date": dates,
            "Open": gift_base * 0.999,
            "High": gift_base * 1.005,
            "Low": gift_base * 0.995,
            "Close": gift_base,
            "Volume": np.zeros(periods),
        }
    )

    changed_gift_df = gift_df.copy()
    changed_gift_df.loc[125:, "Close"] = changed_gift_df.loc[125:, "Close"] * 1.25
    changed_gift_df.loc[125:, "High"] = changed_gift_df.loc[125:, "High"] * 1.25
    changed_gift_df.loc[125:, "Low"] = changed_gift_df.loc[125:, "Low"] * 1.25

    original = build_gift_model_frame(stock_df, gift_df)
    changed = build_gift_model_frame(stock_df, changed_gift_df)
    compare_columns = ["Date", *GIFT_MODEL_FEATURE_COLUMNS]

    original_slice = original.loc[
        original["Date"] <= dates[110],
        compare_columns,
    ].reset_index(drop=True)
    changed_slice = changed.loc[
        changed["Date"] <= dates[110],
        compare_columns,
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(original_slice, changed_slice)
