import numpy as np
import pandas as pd

from src.validation.walk_forward import WalkForwardValidator


class OneDimensionalProbabilityModel:
    def train(self, X, y):
        self._train_size = len(X)

    def predict_proba(self, X):
        return np.linspace(0.40, 0.70, len(X))


def test_walk_forward_validator_supports_one_dimensional_probabilities():
    df = pd.DataFrame({
        "feature": np.arange(10, dtype=float),
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    validator = WalkForwardValidator(
        model=OneDimensionalProbabilityModel(),
        feature_columns=["feature"],
        target_column="target",
        train_window=4,
        step_size=2,
    )

    signals = validator.run(df)

    assert len(signals) == len(df)
    assert set(signals.unique()) <= {0, 1}
