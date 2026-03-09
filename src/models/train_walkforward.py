# src/models/train_walkforward.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from src.repositories.market_data_repository import MarketDataRepository

from src.analytics.features import (
    add_log_returns,
    add_sma,
    add_rsi,
    add_momentum,
    add_volatility,
    add_macd,
    add_atr,
    add_volume_features,
    add_regime_features,
    add_trend_features,
)

from src.backtesting.engine import BacktestEngine
from src.analytics.performance import sharpe_ratio


ARTIFACTS = Path("artifacts/models")
ARTIFACTS.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "log_return",
    "sma_20",
    "sma_50",
    "rsi",
    "momentum_10",
    "volatility_20",
    "macd_hist",
    "atr_14",
    "vol_sma_5",
    "roll_skew_20",
    "roll_kurt_20",
    "sma_diff",
    "sma_ratio",
]


# =========================
# Feature builder
# =========================

def build_features(df):

    df = add_log_returns(df)

    df = add_sma(df, 20)
    df = add_sma(df, 50)

    df = add_rsi(df)
    df = add_momentum(df, 10)
    df = add_volatility(df, 20)

    df = add_macd(df)
    df = add_atr(df)

    df = add_volume_features(df)
    df = add_regime_features(df)

    df = add_trend_features(df)

    df = df.dropna()

    df["target"] = (df["log_return"].shift(-1) > 0).astype(int)

    df = df.dropna()

    return df


# =========================
# Threshold selection
# =========================

def choose_threshold_by_sharpe(model, X_val, df_val, engine):

    prob = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.50, 0.90, 17)

    best_sharpe = -np.inf
    best_threshold = 0.55

    for t in thresholds:

        signals = pd.Series(
            (prob > t).astype(int),
            index=df_val.index,
        )

        bt = engine.run(df_val, signals)

        returns = bt["strategy_return"].dropna()

        if len(returns) < 5:
            continue

        s = sharpe_ratio(returns)

        if s > best_sharpe:
            best_sharpe = s
            best_threshold = t

    return best_threshold, best_sharpe


# =========================
# Training per ticker
# =========================

def train_ticker(ticker, config):

    print(f"Training {ticker}")

    repo = MarketDataRepository()

    df = repo.load(ticker)

    df = build_features(df)

    if df.empty or len(df) < 250:
        print("Not enough training data")
        return None

    engine = BacktestEngine(
        transaction_cost=config.get("transaction_cost", 0.001)
    )

    tscv = TimeSeriesSplit(n_splits=5)

    C_grid = [0.01, 0.1, 1.0, 5.0, 10.0]

    best_meta = None

    for C in C_grid:

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="liblinear",
                    ),
                ),
            ]
        )

        calibrated = CalibratedClassifierCV(
            estimator=pipeline,
            cv=TimeSeriesSplit(3),
        )

        sharpe_scores = []
        thresholds = []

        for train_idx, val_idx in tscv.split(df):

            df_train = df.iloc[train_idx]
            df_val = df.iloc[val_idx]

            X_train = df_train[FEATURES]
            y_train = df_train["target"]

            X_val = df_val[FEATURES]

            calibrated.fit(X_train, y_train)

            t_opt, s_opt = choose_threshold_by_sharpe(
                calibrated,
                X_val,
                df_val,
                engine,
            )

            sharpe_scores.append(s_opt)
            thresholds.append(t_opt)

        avg_sharpe = np.nanmean(sharpe_scores)

        avg_threshold = float(np.nanmean(thresholds))

        if best_meta is None or avg_sharpe > best_meta["avg_sharpe"]:

            best_meta = {
                "C": C,
                "avg_sharpe": avg_sharpe,
                "opt_threshold": avg_threshold,
            }

    print("Best meta:", best_meta)

    # =========================
    # Final training
    # =========================

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=best_meta["C"],
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                ),
            ),
        ]
    )

    final_model = CalibratedClassifierCV(
        estimator=final_pipeline,
        cv=TimeSeriesSplit(3),
    )

    final_model.fit(df[FEATURES], df["target"])

    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    model_name = f"{ticker.replace('.','_')}_logreg_{now}.joblib"

    dump(final_model, ARTIFACTS / model_name)

    meta = {
        "ticker": ticker,
        "features": FEATURES,
        "best_C": best_meta["C"],
        "opt_threshold": best_meta["opt_threshold"],
        "avg_sharpe_cv": best_meta["avg_sharpe"],
        "artifact": str(ARTIFACTS / model_name),
        "trained_on": now,
    }

    meta_path = ARTIFACTS / f"{ticker.replace('.','_')}_meta_{now}.json"

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {model_name}")

    return meta


# =========================
# Run all tickers
# =========================

def run_all():

    config = {"transaction_cost": 0.001}

    repo = MarketDataRepository()

    tickers = repo.list_tickers()

    results = []

    for ticker in tickers:

        meta = train_ticker(ticker, config)

        if meta:
            results.append(meta)

    summary_path = ARTIFACTS / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_all()