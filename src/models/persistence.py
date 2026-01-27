from pathlib import Path
from datetime import datetime
import joblib


ARTIFACT_DIR = Path("artifacts/models")


def save_model(model, model_name: str) -> Path:
    """
    Persist a trained model with versioning.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = ARTIFACT_DIR / f"{model_name}_{timestamp}.joblib"

    joblib.dump(model, path)
    return path


def load_model(path: Path):
    """
    Load a persisted model.
    """
    return joblib.load(path)
