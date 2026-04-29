import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    "log_return",
    "sma_20",
    "sma_50",
    "rsi",
    "momentum_10",
    "volatility_20",
    "macd_hist",
    "atr_14",
    "vol_sma_5",
    "roll_skew_20",
    "roll_kurt_20",
    "sma_diff",
    "sma_ratio",
    "range_pct",
    "intraday_return",
    "overnight_gap",
    "atr_pct",
    "dist_sma_20",
    "dist_sma_50",
    "roll_return_5",
    "roll_return_20",
    "volume_z_20",
    "trend_strength",
]


def add_log_returns(df):

    df=df.copy()

    df["log_return"]=np.log(
        df["Close"]/df["Close"].shift(1)
    )

    return df


def add_sma(df,window):

    df=df.copy()

    df[f"sma_{window}"]=(
        df["Close"].rolling(window).mean()
    )

    return df


def add_volatility(df,window):

    df=df.copy()

    df[f"volatility_{window}"]=(
        df["log_return"].rolling(window).std()
    )

    return df


def add_rsi(df,window=14):

    delta=df["Close"].diff()

    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)

    avg_gain=gain.rolling(window).mean()
    avg_loss=loss.rolling(window).mean()

    rs=avg_gain/avg_loss

    df["rsi"]=100-(100/(1+rs))

    return df


def add_momentum(df,window=10):

    df[f"momentum_{window}"]=(
        df["Close"]/df["Close"].shift(window)-1
    )

    return df


def add_macd(df,fast=12,slow=26,signal=9):

    ema_fast=df["Close"].ewm(
        span=fast,
        adjust=False
    ).mean()

    ema_slow=df["Close"].ewm(
        span=slow,
        adjust=False
    ).mean()

    macd=ema_fast-ema_slow

    macd_signal=macd.ewm(
        span=signal,
        adjust=False
    ).mean()

    df["macd"]=macd
    df["macd_signal"]=macd_signal
    df["macd_hist"]=macd-macd_signal

    return df


def add_atr(df,window=14):

    high=df["High"]
    low=df["Low"]
    close=df["Close"]

    prev_close=close.shift(1)

    tr=pd.concat([

        (high-low).abs(),
        (high-prev_close).abs(),
        (low-prev_close).abs()

    ],axis=1).max(axis=1)

    df["atr_14"]=tr.rolling(window).mean()

    return df


def add_volume_features(df):

    df["vol_change"]=df["Volume"].pct_change()

    df["vol_sma_5"]=(
        df["vol_change"].rolling(5).mean()
    )

    return df


def add_regime_features(df):

    df["roll_skew_20"]=(
        df["log_return"].rolling(20).skew()
    )

    df["roll_kurt_20"]=(
        df["log_return"].rolling(20).kurt()
    )

    return df


def add_trend_features(df):

    df["sma_diff"]=(
        df["sma_20"]-df["sma_50"]
    )

    df["sma_ratio"]=(
        df["sma_20"]/df["sma_50"]
    )

    return df


def add_price_action_features(df):

    close=df["Close"].replace(0,np.nan)
    open_price=df["Open"].replace(0,np.nan)

    df["range_pct"]=(
        (df["High"]-df["Low"])
        /
        close
    )

    df["intraday_return"]=(
        df["Close"]/open_price
    )-1

    df["overnight_gap"]=(
        df["Open"]/df["Close"].shift(1).replace(0,np.nan)
    )-1

    df["atr_pct"]=(
        df["atr_14"]/close
    )

    return df


def add_relative_trend_features(df):

    close=df["Close"].replace(0,np.nan)

    df["dist_sma_20"]=(
        df["Close"]/df["sma_20"]
    )-1

    df["dist_sma_50"]=(
        df["Close"]/df["sma_50"]
    )-1

    df["trend_strength"]=(
        df["sma_diff"]/close
    )

    return df


def add_return_window_features(df):

    df["roll_return_5"]=(
        df["log_return"].rolling(5).sum()
    )

    df["roll_return_20"]=(
        df["log_return"].rolling(20).sum()
    )

    return df


def add_volume_zscore(df,window=20):

    volume_mean=df["Volume"].rolling(window).mean()
    volume_std=df["Volume"].rolling(window).std()

    df[f"volume_z_{window}"]=(
        df["Volume"]-volume_mean
    )/volume_std

    return df


def build_feature_frame(df, dropna=True):

    df=df.copy()

    df=add_log_returns(df)

    df=add_sma(df,20)
    df=add_sma(df,50)

    df=add_rsi(df)

    df=add_momentum(df,10)

    df=add_volatility(df,20)

    df=add_macd(df)

    df=add_atr(df)

    df=add_volume_features(df)

    df=add_regime_features(df)

    df=add_trend_features(df)

    df=add_price_action_features(df)

    df=add_relative_trend_features(df)

    df=add_return_window_features(df)

    df=add_volume_zscore(df,20)

    df=df.replace([np.inf,-np.inf],np.nan)

    if dropna:
        return df.dropna()

    return df
