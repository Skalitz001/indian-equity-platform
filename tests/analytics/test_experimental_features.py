import numpy as np
import pandas as pd

from src.analytics.experimental_features import (
    BASELINE_FEATURE_GROUP,
    DEFAULT_CLASSIFICATION_FEATURE_GROUP,
    DEFAULT_TRADING_FEATURE_GROUP,
    EXPERIMENTAL_FEATURE_COLUMNS,
    MARKET_CONTEXT_FEATURE_COLUMNS,
    MARKET_CONTEXT_FEATURE_GROUPS,
    PAPER_CHANGEPOINT_FEATURE_GROUP,
    PAPER_CHANGEPOINT_MODEL_FEATURE_COLUMNS,
    build_main_feature_frame,
    build_experimental_feature_frame,
    infer_main_feature_group,
    resolve_main_feature_columns,
    resolve_main_feature_group,
)
from src.analytics.changepoint_features import PAPER_CHANGEPOINT_FEATURE_COLUMNS


def test_build_experimental_feature_frame_contains_context_features():
    periods = 120
    base = np.linspace(100, 145, periods)

    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=periods, freq="B"),
        "Open": base * 0.997,
        "High": base * 1.012,
        "Low": base * 0.988,
        "Close": base,
        "Volume": np.linspace(1_000_000, 1_350_000, periods),
    })

    out = build_experimental_feature_frame(df)

    assert set(MARKET_CONTEXT_FEATURE_COLUMNS).issubset(out.columns)
    assert set(EXPERIMENTAL_FEATURE_COLUMNS).issubset(out.columns)
    assert out[EXPERIMENTAL_FEATURE_COLUMNS].isnull().sum().sum() == 0


def test_market_context_feature_groups_cover_all_context_columns():
    grouped_columns = set()

    for columns in MARKET_CONTEXT_FEATURE_GROUPS.values():
        grouped_columns.update(columns)

    assert set(MARKET_CONTEXT_FEATURE_COLUMNS).issubset(grouped_columns)


def test_resolve_main_feature_columns_and_group_round_trip():
    columns = resolve_main_feature_columns(DEFAULT_CLASSIFICATION_FEATURE_GROUP)

    assert set(MARKET_CONTEXT_FEATURE_COLUMNS).issubset(columns)
    assert infer_main_feature_group(columns) == DEFAULT_CLASSIFICATION_FEATURE_GROUP
    assert resolve_main_feature_group(
        features=columns,
    ) == DEFAULT_CLASSIFICATION_FEATURE_GROUP

    paper_columns = resolve_main_feature_columns(DEFAULT_TRADING_FEATURE_GROUP)

    assert DEFAULT_TRADING_FEATURE_GROUP == PAPER_CHANGEPOINT_FEATURE_GROUP
    assert set(PAPER_CHANGEPOINT_FEATURE_COLUMNS).issubset(paper_columns)
    assert paper_columns == PAPER_CHANGEPOINT_MODEL_FEATURE_COLUMNS
    assert infer_main_feature_group(paper_columns) == PAPER_CHANGEPOINT_FEATURE_GROUP


def test_build_main_feature_frame_supports_baseline_and_context_modes():
    periods = 120
    base = np.linspace(100, 145, periods)

    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=periods, freq="B"),
        "Open": base * 0.997,
        "High": base * 1.012,
        "Low": base * 0.988,
        "Close": base,
        "Volume": np.linspace(1_000_000, 1_350_000, periods),
    })

    baseline = build_main_feature_frame(
        df,
        feature_group=BASELINE_FEATURE_GROUP,
    )
    context = build_main_feature_frame(
        df,
        feature_group=DEFAULT_CLASSIFICATION_FEATURE_GROUP,
    )
    paper = build_main_feature_frame(
        df,
        feature_group=DEFAULT_TRADING_FEATURE_GROUP,
    )

    assert len(context.columns) > len(baseline.columns)
    assert len(paper.columns) > len(context.columns)
    assert set(resolve_main_feature_columns(BASELINE_FEATURE_GROUP)).issubset(
        baseline.columns
    )
    assert set(resolve_main_feature_columns(DEFAULT_CLASSIFICATION_FEATURE_GROUP)).issubset(
        context.columns
    )
    assert set(resolve_main_feature_columns(PAPER_CHANGEPOINT_FEATURE_GROUP)).issubset(
        paper.columns
    )
