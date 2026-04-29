import numpy as np
import pandas as pd

from src.analytics.changepoint_features import (
    PAPER_CHANGEPOINT_FEATURE_COLUMNS,
    add_changepoint_features,
)
from src.analytics.features import FEATURE_COLUMNS, build_feature_frame


BASELINE_FEATURE_GROUP = "baseline"
PAPER_CHANGEPOINT_FEATURE_GROUP = "paper_changepoint"

MARKET_CONTEXT_FEATURE_COLUMNS = [
    "body_to_range",
    "wick_imbalance",
    "close_location_20",
    "breakout_distance_20",
    "efficiency_ratio_10",
    "return_hit_rate_10",
    "volatility_ratio_5_20",
    "range_ratio_5_20",
    "volume_ratio_5_20",
    "gap_follow_through",
    "downside_vol_ratio_20",
    "momentum_accel_5_20",
]

DEFAULT_CLASSIFICATION_FEATURE_GROUP = "all_context"
DEFAULT_TRADING_FEATURE_GROUP = PAPER_CHANGEPOINT_FEATURE_GROUP

MARKET_CONTEXT_FEATURE_GROUPS = {
    "structure": [
        "body_to_range",
        "wick_imbalance",
        "gap_follow_through",
    ],
    "regime_breakout": [
        "close_location_20",
        "breakout_distance_20",
        "efficiency_ratio_10",
        "return_hit_rate_10",
        "momentum_accel_5_20",
    ],
    "activity_risk": [
        "volatility_ratio_5_20",
        "range_ratio_5_20",
        "volume_ratio_5_20",
        "downside_vol_ratio_20",
    ],
    "all_context": MARKET_CONTEXT_FEATURE_COLUMNS,
}

MAIN_FEATURE_GROUPS = {
    BASELINE_FEATURE_GROUP: [],
    **MARKET_CONTEXT_FEATURE_GROUPS,
    PAPER_CHANGEPOINT_FEATURE_GROUP: [
        *MARKET_CONTEXT_FEATURE_COLUMNS,
        *PAPER_CHANGEPOINT_FEATURE_COLUMNS,
    ],
}

MAIN_FEATURE_GROUP_CHOICES = (
    *MAIN_FEATURE_GROUPS.keys(),
)

EXPERIMENTAL_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
    *MARKET_CONTEXT_FEATURE_COLUMNS,
]

PAPER_CHANGEPOINT_MODEL_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
    *MAIN_FEATURE_GROUPS[PAPER_CHANGEPOINT_FEATURE_GROUP],
]


def resolve_main_feature_columns(feature_group: str = DEFAULT_TRADING_FEATURE_GROUP) -> list[str]:
    if feature_group == BASELINE_FEATURE_GROUP:
        return list(FEATURE_COLUMNS)

    if feature_group in MAIN_FEATURE_GROUPS:
        return [
            *FEATURE_COLUMNS,
            *MAIN_FEATURE_GROUPS[feature_group],
        ]

    raise ValueError(f"Unsupported main feature group '{feature_group}'")


def infer_main_feature_group(
    features: list[str] | tuple[str, ...] | None,
    fallback: str = DEFAULT_TRADING_FEATURE_GROUP,
) -> str:
    if not features:
        return fallback

    feature_list = list(features)

    if feature_list == list(FEATURE_COLUMNS):
        return BASELINE_FEATURE_GROUP

    for feature_group in MAIN_FEATURE_GROUPS:
        if feature_list == resolve_main_feature_columns(feature_group):
            return feature_group

    return fallback


def resolve_main_feature_group(
    meta: dict | None = None,
    features: list[str] | tuple[str, ...] | None = None,
    fallback: str = DEFAULT_TRADING_FEATURE_GROUP,
) -> str:
    if isinstance(meta, dict) and meta.get("feature_group") in MAIN_FEATURE_GROUP_CHOICES:
        return str(meta["feature_group"])

    return infer_main_feature_group(features, fallback=fallback)


def build_experimental_feature_frame(
    df: pd.DataFrame,
    dropna: bool = True,
) -> pd.DataFrame:
    frame = build_feature_frame(df, dropna=False).copy()

    high = frame["High"].replace(0, np.nan)
    low = frame["Low"].replace(0, np.nan)
    close = frame["Close"].replace(0, np.nan)
    open_price = frame["Open"].replace(0, np.nan)
    daily_range = (high - low).replace(0, np.nan)

    candle_high = pd.concat(
        [frame["Open"], frame["Close"]],
        axis=1,
    ).max(axis=1)
    candle_low = pd.concat(
        [frame["Open"], frame["Close"]],
        axis=1,
    ).min(axis=1)

    upper_wick = high - candle_high
    lower_wick = candle_low - low

    frame["body_to_range"] = (
        frame["Close"] - frame["Open"]
    ) / daily_range
    frame["wick_imbalance"] = (
        lower_wick - upper_wick
    ) / daily_range

    prior_high_20 = high.shift(1).rolling(20).max()
    prior_low_20 = low.shift(1).rolling(20).min()
    prior_range_20 = (prior_high_20 - prior_low_20).replace(0, np.nan)

    frame["close_location_20"] = (
        frame["Close"] - prior_low_20
    ) / prior_range_20
    frame["breakout_distance_20"] = (
        frame["Close"] / prior_high_20
    ) - 1

    directionless_move_10 = close.diff().abs().rolling(10).sum().replace(0, np.nan)
    frame["efficiency_ratio_10"] = (
        (frame["Close"] - frame["Close"].shift(10)).abs()
        / directionless_move_10
    )
    frame["return_hit_rate_10"] = (
        (frame["log_return"] > 0).astype(float).rolling(10).mean()
    )

    volatility_5 = frame["log_return"].rolling(5).std()
    frame["volatility_ratio_5_20"] = (
        volatility_5 / frame["volatility_20"].replace(0, np.nan)
    )

    short_range = frame["range_pct"].rolling(5).mean()
    long_range = frame["range_pct"].rolling(20).mean().replace(0, np.nan)
    frame["range_ratio_5_20"] = short_range / long_range

    short_volume = frame["Volume"].rolling(5).mean()
    long_volume = frame["Volume"].rolling(20).mean().replace(0, np.nan)
    frame["volume_ratio_5_20"] = (short_volume / long_volume) - 1

    frame["gap_follow_through"] = (
        np.sign(frame["overnight_gap"].fillna(0.0))
        * frame["intraday_return"]
    )

    downside_volatility_20 = (
        frame["log_return"].clip(upper=0).pow(2).rolling(20).mean().pow(0.5)
    )
    frame["downside_vol_ratio_20"] = (
        downside_volatility_20
        / frame["volatility_20"].replace(0, np.nan)
    )

    frame["momentum_accel_5_20"] = (
        frame["roll_return_5"] - (frame["roll_return_20"] / 4.0)
    )

    frame = frame.replace([np.inf, -np.inf], np.nan)

    if dropna:
        return frame.dropna(subset=EXPERIMENTAL_FEATURE_COLUMNS)

    return frame


def build_main_feature_frame(
    df: pd.DataFrame,
    feature_group: str = DEFAULT_TRADING_FEATURE_GROUP,
    dropna: bool = True,
) -> pd.DataFrame:
    feature_columns = resolve_main_feature_columns(feature_group)

    if feature_group == BASELINE_FEATURE_GROUP:
        return build_feature_frame(df, dropna=dropna)

    frame = build_experimental_feature_frame(df, dropna=False)

    if feature_group == PAPER_CHANGEPOINT_FEATURE_GROUP:
        frame = add_changepoint_features(frame)

    if dropna:
        return frame.dropna(subset=feature_columns)

    return frame
