import numpy as np
import pandas as pd


def probability_down(probability_up):
    if pd.isna(probability_up):
        return np.nan

    return 1.0 - float(probability_up)


def prediction_label(probability_up, threshold=0.50):
    return "UP" if probability_up >= threshold else "DOWN"


def prediction_probability(probability_up, prediction):
    if pd.isna(probability_up) or prediction == "N/A":
        return np.nan

    if prediction == "DOWN":
        return probability_down(probability_up)

    return float(probability_up)


def prediction_probability_label(prediction):
    if prediction == "DOWN":
        return "P(Down)"

    if prediction == "UP":
        return "P(Up)"

    return "Probability"


def format_probability(value):
    return "N/A" if pd.isna(value) else f"{float(value):.2%}"


def format_prediction_probability(probability_up, prediction, separator=": "):
    label = prediction_probability_label(prediction)
    value = prediction_probability(probability_up, prediction)

    return f"{label}{separator}{format_probability(value)}"


def format_prediction_probability_markdown(probability_up, prediction):
    label = prediction_probability_label(prediction)
    value = prediction_probability(probability_up, prediction)

    return f"{label}: `{format_probability(value)}`"
