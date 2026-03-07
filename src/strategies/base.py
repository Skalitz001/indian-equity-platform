from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Base interface for all trading strategies.
    """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns a Series of positions (0 or 1).
        """
        pass
