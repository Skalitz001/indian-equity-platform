import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from src.analytics.experimental_features import (
    build_main_feature_frame,
    resolve_main_feature_group,
)
from src.models.probabilities import predict_up_probability
from src.models.persistence import load_model
from src.repositories.market_data_repository import MarketDataRepository
from src.strategies.signal_policy import normalize_signal_policy


ARTIFACTS_DIR = Path("artifacts/models")


app = FastAPI(title="Indian Equity Signal API")

repo = MarketDataRepository()


def load_meta(ticker: str) -> dict:

    meta_path=ARTIFACTS_DIR/f"{ticker.replace('.','_')}_meta.json"

    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained model metadata found for ticker '{ticker}'",
        )

    return json.loads(meta_path.read_text())


def load_prediction_model(meta: dict, model_name: str):

    log_path=ARTIFACTS_DIR/meta["log_model"]
    rf_path=ARTIFACTS_DIR/meta["rf_model"]

    if model_name=="logistic":
        return load_model(log_path)

    if model_name=="rf":
        return load_model(rf_path)

    if model_name=="ensemble":
        return {
            "kind":"ensemble",
            "log_model":load_model(log_path),
            "rf_model":load_model(rf_path),
            "weights":meta.get("blend_weights",{
                "logistic":0.5,
                "rf":0.5,
            }),
        }

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported model '{model_name}'",
    )

@app.get("/")
def root():
    return {
        "service": "Indian Equity Signal API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(
    ticker: str = Query(
        ...,
        description="NSE ticker symbol (e.g. RELIANCE.NS)",
        example="RELIANCE.NS",
        min_length=3,
        max_length=20,
        regex=r"^[A-Z0-9^]+\.NS$",
    ),
    model_name: str = Query(
        "logistic",
        description="Model to use: logistic, rf, or ensemble",
        pattern="^(logistic|rf|ensemble)$",
    ),
):
    try:
        df = repo.load(ticker)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for ticker '{ticker}'",
        )

    meta=load_meta(ticker)

    if model_name=="ensemble" and "blend_weights" not in meta:
        model_name="logistic"

    model=load_prediction_model(meta,model_name)

    feature_group=resolve_main_feature_group(
        meta=meta,
        features=meta.get("features"),
    )
    feature_df=build_main_feature_frame(
        df,
        feature_group=feature_group,
    )

    if feature_df.empty:
        raise HTTPException(
            status_code=422,
            detail="Not enough data to generate prediction",
        )

    features=meta["features"]

    missing_features=[
        feature for feature in features
        if feature not in feature_df.columns
    ]

    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=(
                "Feature mismatch between metadata and inference frame: "
                f"{missing_features}"
            ),
        )

    probability_up=float(
        predict_up_probability(
            model,
            feature_df[features].tail(1),
        )[0]
    )

    signal_policy=normalize_signal_policy(
        meta.get("signal_policies",{}).get(model_name),
        fallback_threshold=float(
            meta.get("thresholds",{}).get(
                model_name,
                meta.get("opt_threshold",0.55),
            )
        ),
    )

    threshold=float(signal_policy["entry_threshold"])
    signal=int(probability_up>threshold)

    return {
        "ticker": ticker,
        "model": model_name,
        "probability_up": round(probability_up, 4),
        "threshold": round(threshold, 4),
        "exit_threshold": round(float(signal_policy["exit_threshold"]), 4),
        "signal": signal,
        "action": "BUY" if signal == 1 else "HOLD",
        "recommended_model": meta.get("recommended_model"),
        "feature_group": feature_group,
    }
