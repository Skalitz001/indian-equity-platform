from pathlib import Path

import joblib


def load_model(path: Path):
    """
    Load a persisted model.
    """
    return joblib.load(path)
