import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd

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

from src.models.persistence import load_model

from src.backtesting.engine import BacktestEngine

from src.analytics.performance import (
    total_return,
    annualized_return,
    sharpe_ratio,
    max_drawdown,
    win_rate,
)

from src.validation.metrics import classification_metrics


# =========================
# App configuration
# =========================

st.set_page_config(
    page_title="Indian Equity ML Signals",
    layout="wide",
)

st.title("Indian Equity ML Signal Dashboard")
st.caption("End-to-end analytics + ML platform (NSE stocks)")


# =========================
# Initialize shared objects
# =========================

repo = MarketDataRepository()
engine = BacktestEngine(transaction_cost=0.001)

model_files = list(Path("artifacts/models").glob("*.joblib"))

if not model_files:
    st.error("No trained models found. Run training first.")
    st.stop()

MODEL_PATH = max(model_files)

model = load_model(MODEL_PATH)


# =========================
# Feature list
# =========================

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
# Sidebar controls
# =========================

st.sidebar.header("Configuration")

tickers = repo.list_tickers()

if not tickers:
    st.error("No tickers found. Run ingestion first.")
    st.stop()

ticker = st.sidebar.selectbox(
    "Select NSE Ticker",
    tickers
)

threshold = st.sidebar.slider(
    "Buy Probability Threshold",
    min_value=0.50,
    max_value=0.75,
    value=0.55,
    step=0.01,
)


# =========================
# Load data
# =========================

df = repo.load(ticker)


# =========================
# Feature engineering
# =========================

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

df["bh_return"] = df["log_return"]
df["bh_equity"] = (1 + df["bh_return"]).cumprod()

df = df.dropna()

if df.empty:
    st.error("Not enough data after feature engineering.")
    st.stop()


# =========================
# ML target
# =========================

df["target"] = (df["log_return"].shift(-1) > 0).astype(int)

df = df.dropna()


# =========================
# Predictions
# =========================

proba = model.predict_proba(df[FEATURES])[:, 1]

signals = pd.Series(
    (proba > threshold).astype(int),
    index=df.index
)
preds = signals.values

trade_mask = signals == 1


# =========================
# Metrics
# =========================

if trade_mask.sum() > 0:

    cls_metrics = classification_metrics(
        df.loc[trade_mask, "target"],
        preds[trade_mask],
    )

else:

    cls_metrics = {
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0,
    }


# =========================
# Latest signal
# =========================

latest_proba = proba[-1]
latest_signal = int(latest_proba > threshold)

c1, c2, c3 = st.columns(3)

c1.metric(
    "Latest Close Price",
    f"₹{df['Close'].iloc[-1]:,.2f}",
)

c2.metric(
    "Probability Up",
    f"{latest_proba:.2%}",
)

c3.metric(
    "Signal",
    "BUY" if latest_signal else "HOLD",
)


# =========================
# Price chart
# =========================

st.subheader("Price Chart")

price_fig = go.Figure()

price_fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        name="Close Price",
        line=dict(width=2),
    )
)

price_fig.update_layout(height=400)

st.plotly_chart(price_fig, use_container_width=True)


# =========================
# Backtest
# =========================

bt = engine.run(df, signals)

returns = bt["strategy_return"].dropna()

equity = bt["equity_curve"]


# =========================
# Strategy metrics
# =========================

st.subheader("Strategy Performance Metrics")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Total Return",
    f"{total_return(equity):.2%}",
)

m2.metric(
    "Annualized Return",
    f"{annualized_return(returns):.2%}",
)

m3.metric(
    "Sharpe Ratio",
    f"{sharpe_ratio(returns):.2f}",
)

m4.metric(
    "Max Drawdown",
    f"{max_drawdown(equity):.2%}",
)

m5.metric(
    "Win Rate",
    f"{win_rate(returns):.2%}",
)


# =========================
# Equity curve
# =========================

st.subheader("Strategy Equity Curve")

equity_fig = go.Figure()

equity_fig.add_trace(
    go.Scatter(
        x=bt["Date"],
        y=bt["equity_curve"],
        name="ML Strategy",
    )
)

equity_fig.update_layout(height=400)

st.plotly_chart(equity_fig, use_container_width=True)


# =========================
# ML validation metrics
# =========================

st.subheader("ML Validation Metrics")

v1, v2, v3, v4 = st.columns(4)

v1.metric("Accuracy", f"{cls_metrics['accuracy']:.2%}")
v2.metric("Precision", f"{cls_metrics['precision']:.2%}")
v3.metric("Recall", f"{cls_metrics['recall']:.2%}")
v4.metric("F1 Score", f"{cls_metrics['f1']:.2%}")


# =========================
# Buy & Hold benchmark
# =========================

bh_returns = df["bh_return"].dropna()
bh_equity = df["bh_equity"]

st.subheader("Strategy vs Buy & Hold")

b1, b2, b3, b4 = st.columns(4)

b1.metric(
    "Strategy Sharpe",
    f"{sharpe_ratio(returns):.2f}",
)

b2.metric(
    "Buy & Hold Sharpe",
    f"{sharpe_ratio(bh_returns):.2f}",
)

b3.metric(
    "Strategy Max DD",
    f"{max_drawdown(equity):.2%}",
)

b4.metric(
    "Buy & Hold Max DD",
    f"{max_drawdown(bh_equity):.2%}",
)


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=equity,
        name="ML Strategy",
    )
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=bh_equity,
        name="Buy & Hold",
    )
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# Global ML metrics
# =========================

st.subheader("ML Metrics (Global)")

global_metrics = classification_metrics(
    df["target"],
    preds,
)

st.write(global_metrics)


# =========================
# Trade precision
# =========================

st.subheader("ML Metrics (Trade Conditional)")

if trade_mask.sum() > 0:

    trade_precision = (
        df.loc[trade_mask, "target"] == 1
    ).mean()

else:

    trade_precision = 0


st.metric(
    "Trade Precision",
    f"{trade_precision:.2%}",
)


# =========================
# Footer
# =========================

st.caption(
    "⚠️ Educational use only. Signals are generated from historical data."
)