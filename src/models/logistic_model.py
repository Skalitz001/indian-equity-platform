import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class DirectionLogisticModel:
    """
    Binary classifier for next-day direction.
    """

    def __init__(self):
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        )

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Returns probability of upward move.
        """
        return self.pipeline.predict_proba(X)[:, 1]
