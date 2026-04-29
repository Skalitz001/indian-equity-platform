import numpy as np


def positive_class_probabilities(raw_proba) -> np.ndarray:
    probabilities = np.asarray(raw_proba, dtype=float)

    if probabilities.ndim == 1:
        return probabilities

    return probabilities[:, 1]


def predict_up_probability(model, X) -> np.ndarray:
    if isinstance(model, dict) and model.get("kind") == "ensemble":
        weights = model["weights"]

        return (
            float(weights["logistic"])
            * predict_up_probability(model["log_model"], X)
            + float(weights["rf"])
            * predict_up_probability(model["rf_model"], X)
        )

    return positive_class_probabilities(model.predict_proba(X))
