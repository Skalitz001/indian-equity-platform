import pandas as pd
from src.backtesting.engine import BacktestEngine


def test_backtest_runs():
    df = pd.DataFrame({
        "log_return": [0.01, -0.02, 0.03]
    })
    signal = pd.Series([1, 1, 0])

    engine = BacktestEngine()
    result = engine.run(df, signal)

    assert "equity_curve" in result.columns
    assert result["equity_curve"].iloc[0] == 1.0
