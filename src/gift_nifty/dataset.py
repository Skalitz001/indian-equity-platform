import numpy as np
import pandas as pd

from src.analytics.experimental_features import (
    DEFAULT_TRADING_FEATURE_GROUP,
    build_main_feature_frame,
    resolve_main_feature_columns,
)
from src.gift_nifty.constants import GIFT_NIFTY_START_DATE
from src.gift_nifty.features import GIFT_FEATURE_COLUMNS, build_gift_feature_frame


GIFT_STOCK_FEATURE_GROUP = DEFAULT_TRADING_FEATURE_GROUP
GIFT_STOCK_FEATURE_COLUMNS = resolve_main_feature_columns(GIFT_STOCK_FEATURE_GROUP)
GIFT_PRE_OPEN_SOURCE_ASSUMPTION = (
    "Each GIFT Nifty Date row is treated as a completed pre-open Singapore-time "
    "session that is available before the matching NSE cash-market session opens."
)

GIFT_MODEL_FEATURE_COLUMNS = [
    *[f"stock_prev_{column}" for column in GIFT_STOCK_FEATURE_COLUMNS],
    *GIFT_FEATURE_COLUMNS,
    "gift_source_age_days",
]


def build_gift_model_frame(
    stock_df: pd.DataFrame,
    gift_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build same-day intraday stock labels from previous stock state plus GIFT context.

    Contract: same-date GIFT rows are intentionally allowed because the GIFT row is
    assumed to be a pre-open signal available before the matching NSE session.
    Stock-derived model inputs remain shifted by one trading day.
    """
    stock_history = stock_df.copy()
    stock_history["Date"] = pd.to_datetime(
        stock_history["Date"],
        errors="coerce",
    ).dt.normalize()
    stock_history = stock_history.dropna(subset=["Date"])
    stock_history = stock_history.sort_values("Date").reset_index(drop=True)

    stock_features = build_main_feature_frame(
        stock_history,
        feature_group=GIFT_STOCK_FEATURE_GROUP,
        dropna=False,
    )[["Date", *GIFT_STOCK_FEATURE_COLUMNS]].copy()

    renamed_columns = {
        column: f"stock_prev_{column}"
        for column in GIFT_STOCK_FEATURE_COLUMNS
    }

    for column in GIFT_STOCK_FEATURE_COLUMNS:
        stock_features[column] = stock_features[column].shift(1)

    stock_features = stock_features.rename(columns=renamed_columns)

    frame = stock_history.merge(
        stock_features,
        on="Date",
        how="left",
    )

    open_price = frame["Open"].replace(0, np.nan)
    frame["intraday_return"] = (frame["Close"] / open_price) - 1
    frame["target"] = np.where(
        frame["intraday_return"].notna(),
        (frame["intraday_return"] > 0).astype(int),
        np.nan,
    )

    gift_features = build_gift_feature_frame(
        gift_df,
        dropna=False,
    )[["Date", *GIFT_FEATURE_COLUMNS]].copy()
    gift_features = gift_features.rename(
        columns={"Date": "gift_source_date"}
    )

    merged = pd.merge_asof(
        frame.sort_values("Date"),
        gift_features.sort_values("gift_source_date"),
        left_on="Date",
        right_on="gift_source_date",
        direction="backward",
    )

    merged["gift_source_age_days"] = (
        merged["Date"] - merged["gift_source_date"]
    ).dt.days.astype(float)

    merged = merged.loc[
        merged["Date"] >= GIFT_NIFTY_START_DATE
    ].copy()

    merged = merged.replace([np.inf, -np.inf], np.nan)

    required_columns = [
        *GIFT_MODEL_FEATURE_COLUMNS,
        "target",
        "intraday_return",
    ]

    merged = merged.dropna(subset=required_columns)

    return merged.reset_index(drop=True)
