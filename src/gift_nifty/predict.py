import argparse
import json
from pathlib import Path

from src.gift_nifty.constants import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_GIFT_DATA_PATH,
)
from src.gift_nifty.train_walkforward import build_training_frame
from src.models.persistence import load_model
from src.models.probabilities import predict_up_probability


ARTIFACTS = Path(DEFAULT_ARTIFACTS_DIR)


def load_meta(ticker: str) -> dict:
    meta_path = ARTIFACTS / f"{ticker.replace('.','_')}_gift_meta.json"

    if not meta_path.exists():
        raise FileNotFoundError(
            f"No GIFT-aware metadata found for ticker '{ticker}'"
        )

    return json.loads(meta_path.read_text())


def load_prediction_model(meta: dict, model_name: str):
    log_path = ARTIFACTS / meta["log_model"]
    rf_path = ARTIFACTS / meta["rf_model"]

    if model_name == "logistic":
        return load_model(log_path)

    if model_name == "rf":
        return load_model(rf_path)

    if model_name == "ensemble":
        return {
            "kind": "ensemble",
            "log_model": load_model(log_path),
            "rf_model": load_model(rf_path),
            "weights": meta.get("blend_weights", {
                "logistic": 0.5,
                "rf": 0.5,
            }),
        }

    raise ValueError(f"Unsupported model '{model_name}'")


def predict_latest(
    ticker: str,
    model_name: str = "ensemble",
    gift_path: str = DEFAULT_GIFT_DATA_PATH,
) -> dict:
    meta = load_meta(ticker)

    if model_name == "ensemble" and "blend_weights" not in meta:
        model_name = "logistic"

    model = load_prediction_model(meta, model_name)
    frame = build_training_frame(
        ticker=ticker,
        gift_path=gift_path,
    )

    if frame.empty:
        raise ValueError("No GIFT-aligned model frame available for prediction")

    latest_row = frame.iloc[-1]
    features = meta["features"]

    probability_up = float(
        predict_up_probability(
            model,
            frame[features].tail(1),
        )[0]
    )

    threshold = float(
        meta.get("thresholds", {}).get(
            model_name,
            meta.get("opt_threshold", 0.55),
        )
    )

    signal = int(probability_up > threshold)

    return {
        "ticker": ticker,
        "model": model_name,
        "session_date": str(latest_row["Date"].date()),
        "gift_source_date": str(latest_row["gift_source_date"].date()),
        "probability_up": round(probability_up, 4),
        "threshold": round(threshold, 4),
        "signal": signal,
        "action": "BUY_OPEN_SELL_CLOSE" if signal == 1 else "SKIP",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict latest intraday signal from standalone GIFT-aware models",
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument(
        "--model-name",
        default="ensemble",
        choices=["logistic", "rf", "ensemble"],
    )
    parser.add_argument(
        "--gift-path",
        default=DEFAULT_GIFT_DATA_PATH,
        help="Path to normalized post-2023 GIFT Nifty OHLC CSV.",
    )

    args = parser.parse_args()

    print(json.dumps(
        predict_latest(
            ticker=args.ticker,
            model_name=args.model_name,
            gift_path=args.gift_path,
        ),
        indent=2,
    ))
