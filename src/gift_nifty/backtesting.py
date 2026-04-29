import pandas as pd


class IntradayBacktestEngine:
    """
    Long-only intraday backtest for open-to-close execution.

    Signals are assumed to be known before the cash session opens.
    """

    def __init__(self, transaction_cost: float = 0.001):
        self.transaction_cost = float(transaction_cost)

    def run(self, df: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
        if not isinstance(signal, pd.Series):
            raise TypeError(
                "signal must be a pandas Series with index aligned to df"
            )

        result = df.copy()
        result["position"] = signal.astype(float).reindex(result.index).fillna(0.0)

        # Round-trip cost is paid on every active intraday position.
        result["strategy_return"] = (
            result["position"] * result["intraday_return"]
            - (2 * self.transaction_cost * result["position"])
        )
        result["equity_curve"] = (1 + result["strategy_return"]).cumprod()

        return result
