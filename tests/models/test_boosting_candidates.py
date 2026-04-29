import numpy as np
import pandas as pd

from src.models.boosting_candidates import build_adaboost_candidates
from src.models.probabilities import predict_up_probability


def test_build_adaboost_candidates_fit_and_predict_probabilities():
    candidate = build_adaboost_candidates()[0]
    model = candidate["builder"]()

    feature_a = np.concatenate([
        np.linspace(-3.0, -0.1, 20),
        np.linspace(0.1, 3.0, 20),
    ])
    feature_b = np.tile([0.0, 0.1, -0.1, 0.2], 10)

    X = pd.DataFrame({
        "feature_a": feature_a,
        "feature_b": feature_b,
    })
    y = np.array([0] * 20 + [1] * 20)

    model.fit(X, y)
    probabilities = predict_up_probability(model, X)

    assert probabilities.shape == (len(X),)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
