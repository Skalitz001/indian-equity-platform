from fastapi import FastAPI
from pathlib import Path
from fastapi import HTTPException, Query

from src.repositories.market_data_repository import MarketDataRepository
from src.analytics.features import add_log_returns, add_sma
from src.models.persistence import load_model

FEATURES = ["log_return", "sma_20", "sma_50"]
MODEL_PATH = max(Path("artifacts/models").glob("*.joblib"))

app = FastAPI(title="Indian Equity Signal API")

repo = MarketDataRepository()
model = load_model(MODEL_PATH)

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
        regex=r"^[A-Z0-9]+\.NS$",
    )
):
    try:
        df = repo.load(ticker)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for ticker '{ticker}'",
        )

    df = add_log_returns(df)
    df = add_sma(df, 20)
    df = add_sma(df, 50)
    df = df.dropna()

    if df.empty:
        raise HTTPException(
            status_code=422,
            detail="Not enough data to generate prediction",
        )

    proba = model.predict_proba(df[FEATURES])[-1]
    signal = int(proba > 0.55)

    return {
        "ticker": ticker,
        "probability_up": round(float(proba), 4),
        "signal": signal,
        "action": "BUY" if signal == 1 else "HOLD",
    }