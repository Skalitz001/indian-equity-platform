import numpy as np

from src.models.probabilities import (
    positive_class_probabilities,
    predict_up_probability,
)


class OneDimensionalModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        return self.probabilities


class TwoDimensionalModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        return self.probabilities


def test_positive_class_probabilities_handles_1d_and_2d_outputs():
    one_dimensional = positive_class_probabilities([0.7, 0.4])
    two_dimensional = positive_class_probabilities(
        [[0.3, 0.7], [0.6, 0.4]]
    )

    assert one_dimensional.tolist() == [0.7, 0.4]
    assert two_dimensional.tolist() == [0.7, 0.4]


def test_predict_up_probability_blends_ensemble_members():
    ensemble = {
        "kind": "ensemble",
        "weights": {
            "logistic": 0.25,
            "rf": 0.75,
        },
        "log_model": OneDimensionalModel([0.8, 0.3]),
        "rf_model": TwoDimensionalModel(
            [[0.1, 0.9], [0.6, 0.4]]
        ),
    }

    probabilities = predict_up_probability(ensemble, X=None)

    assert np.allclose(probabilities, [0.875, 0.375])
