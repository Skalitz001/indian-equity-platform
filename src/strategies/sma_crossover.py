import pandas as pd
from src.strategies.base import Strategy


class SMACrossoverStrategy(Strategy):
    """
    Long-only SMA crossover strategy.
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")

        self.short = short_window
        self.long = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        short_col = f"sma_{self.short}"
        long_col = f"sma_{self.long}"

        if short_col not in df.columns or long_col not in df.columns:
            raise KeyError("Required SMA columns not found in DataFrame")

        signals = (df[short_col] > df[long_col]).astype(int)
        return signals
