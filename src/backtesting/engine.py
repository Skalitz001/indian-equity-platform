import pandas as pd
import numpy as np


class BacktestEngine:
    """
    Simple long-only backtesting engine.
    """

    def __init__(self, transaction_cost: float = 0.001):
        """
        Parameters
        ----------
        transaction_cost : float
            Cost per trade (e.g., 0.001 = 10 bps)
        """
        self.transaction_cost = transaction_cost

    def run(self, df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
        """
        Run backtest on provided dataframe and signal.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing at least a 'log_return' column.
        signal : pd.Series
            Trading signal aligned with df.index.
            1 = long, 0 = flat.

        Returns
        -------
        pd.DataFrame
            DataFrame with positions, strategy returns, and equity curve.
        """
        if not isinstance(signal, pd.Series):
            raise TypeError(
                "signal must be a pandas Series with index aligned to df"
            )

        result = df.copy()

        # Position enters on next bar (no look-ahead)
        result["position"] = signal.shift(1).fillna(0)

        # Strategy returns
        result["strategy_return"] = (
            result["position"] * result["log_return"]
            - self.transaction_cost
            * result["position"].diff().abs().fillna(0)
        )

        # Equity curve
        result["equity_curve"] = (1 + result["strategy_return"]).cumprod()

        return result

