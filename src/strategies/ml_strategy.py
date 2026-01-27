import pandas as pd
from src.strategies.base import Strategy


class MLStrategy(Strategy):
    """
    ML-based long-only strategy using probability threshold.
    """

    def __init__(self, model, feature_columns, threshold: float = 0.55):
        self.model = model
        self.feature_columns = feature_columns
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_columns]
        proba = self.model.predict_proba(X)
        signals = (proba > self.threshold).astype(int)
        return pd.Series(signals, index=df.index)
