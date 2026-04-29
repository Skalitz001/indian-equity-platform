import numpy as np
import pandas as pd


GIFT_FEATURE_COLUMNS = [
    "gift_log_return",
    "gift_intraday_return",
    "gift_gap_return",
    "gift_range_pct",
    "gift_volatility_5",
    "gift_roll_return_5",
    "gift_roll_return_20",
    "gift_sma_ratio_5_20",
    "gift_dist_sma_5",
    "gift_trend_strength",
]


def build_gift_feature_frame(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    frame = df.copy()

    close = frame["Close"].replace(0, np.nan)
    open_price = frame["Open"].replace(0, np.nan)
    previous_close = frame["Close"].shift(1).replace(0, np.nan)

    frame["gift_log_return"] = np.log(close / previous_close)
    frame["gift_intraday_return"] = (close / open_price) - 1
    frame["gift_gap_return"] = (open_price / previous_close) - 1
    frame["gift_range_pct"] = (frame["High"] - frame["Low"]) / close

    sma_5 = frame["Close"].rolling(5).mean()
    sma_20 = frame["Close"].rolling(20).mean()

    frame["gift_volatility_5"] = frame["gift_log_return"].rolling(5).std()
    frame["gift_roll_return_5"] = frame["gift_log_return"].rolling(5).sum()
    frame["gift_roll_return_20"] = frame["gift_log_return"].rolling(20).sum()
    frame["gift_sma_ratio_5_20"] = sma_5 / sma_20
    frame["gift_dist_sma_5"] = (frame["Close"] / sma_5) - 1
    frame["gift_trend_strength"] = (sma_5 - sma_20) / close

    frame = frame.replace([np.inf, -np.inf], np.nan)

    if dropna:
        return frame.dropna().reset_index(drop=True)

    return frame
