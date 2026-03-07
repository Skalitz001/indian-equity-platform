import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DirectionLogisticModel:
    """
    Logistic regression model for predicting next-period
    return direction (up / down).

    This model is intentionally simple, interpretable,
    and stable for time-series ML.
    """

    def __init__(self):
        """
        Initialize the model pipeline.

        Design choices:
        - StandardScaler: features have different magnitudes
        - class_weight='balanced': handle slight class imbalance
        - lbfgs solver: stable for small/medium feature sets
        """
        self.model_version = "v2"
        

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        n_jobs=None,
                    ),
                ),
            ]
        )
        self.is_trained = False



    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary target (1 = up, 0 = down).
        """
        if X.empty or y.empty:
            raise ValueError("Training data is empty")

        self.pipeline.fit(X, y)
        self.is_trained = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of upward move.
        """
        # Backward compatibility for older saved models
        if not hasattr(self, "is_trained"):
            self.is_trained = True

        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        return self.pipeline.predict_proba(X)[:, 1]


    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels using a probability threshold.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        threshold : float
            Decision threshold.

        Returns
        -------
        np.ndarray
            Binary predictions.
        """
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)

    def get_coefficients(self, feature_names: list[str]) -> pd.Series:
        """
        Return model coefficients for interpretability.

        Parameters
        ----------
        feature_names : list[str]
            Names of input features.

        Returns
        -------
        pd.Series
            Feature coefficients.
        """
        model = self.pipeline.named_steps["model"]

        if not self.is_trained:
            raise RuntimeError("Model must be trained to inspect coefficients")

        return pd.Series(
            model.coef_[0],
            index=feature_names,
            name="logistic_coefficient",
        ).sort_values(ascending=False)
