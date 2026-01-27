import pandas as pd
from src.analytics.features import add_log_returns, add_sma


def test_add_log_returns():
    df = pd.DataFrame({"Close": [100, 101, 102]})
    out = add_log_returns(df)

    assert "log_return" in out.columns
    assert out["log_return"].isna().sum() == 1


def test_add_sma():
    df = pd.DataFrame({"Close": [1, 2, 3, 4, 5]})
    out = add_sma(df, window=3)

    assert "sma_3" in out.columns
    assert out["sma_3"].iloc[2] == 2.0
