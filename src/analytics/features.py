import pandas as pd
import numpy as np


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds log returns to the dataframe.
    """
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_sma(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Adds simple moving average (SMA).
    """
    df = df.copy()
    df[f"sma_{window}"] = df["Close"].rolling(window).mean()
    return df


def add_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Adds rolling volatility of log returns.
    """
    df = df.copy()
    df[f"volatility_{window}"] = df["log_return"].rolling(window).std()
    return df

def add_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def add_momentum(df, window=10):
    df[f"momentum_{window}"] = df["Close"] / df["Close"].shift(window) - 1
    return df


def add_volatility(df, window=20):
    df[f"volatility_{window}"] = df["log_return"].rolling(window).std()
    return df



def add_macd(df, fast=12, slow=26, signal=9):

    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()

    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    return df


def add_atr(df, window=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = tr.rolling(window).mean()

    return df


def add_volume_features(df):

    df["vol_change"] = df["Volume"].pct_change()
    df["vol_sma_5"] = df["vol_change"].rolling(5).mean()

    return df


def add_regime_features(df):

    df["roll_skew_20"] = df["log_return"].rolling(20).skew()
    df["roll_kurt_20"] = df["log_return"].rolling(20).kurt()

    return df


def add_trend_features(df):

    df["sma_diff"] = df["sma_20"] - df["sma_50"]
    df["sma_ratio"] = df["sma_20"] / df["sma_50"]

    return df