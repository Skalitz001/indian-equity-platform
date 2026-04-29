import pandas as pd
from src.strategies.base import Strategy
from src.strategies.signal_policy import generate_probability_signals


class MLStrategy(Strategy):
    """
    ML-based long-only strategy using probability threshold.
    """

    def __init__(
        self,
        model,
        feature_columns,
        threshold: float = 0.55,
        exit_threshold: float | None = None,
    ):
        self.model = model
        self.feature_columns = feature_columns
        self.signal_policy = {
            "entry_threshold": float(threshold),
            "exit_threshold": (
                float(exit_threshold)
                if exit_threshold is not None else float(threshold)
            ),
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_columns]
        proba = self.model.predict_proba(X)

        if getattr(proba, "ndim", 1) > 1:
            proba = proba[:, 1]

        signals = generate_probability_signals(
            pd.Series(proba, index=df.index),
            self.signal_policy,
        )

        return pd.Series(signals, index=df.index)
