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

