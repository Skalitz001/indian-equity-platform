from pathlib import Path
from datetime import datetime
import pandas as pd


SIGNAL_DIR = Path("artifacts/signals")


def save_signals(
    signals: pd.Series,
    ticker: str,
    model_name: str,
) -> Path:
    """
    Persist generated signals with versioning.
    """
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = SIGNAL_DIR / f"{ticker}_{model_name}_{timestamp}.csv"

    signals.to_frame(name="signal").to_csv(path)
    return path
