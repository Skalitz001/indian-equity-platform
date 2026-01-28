import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd

from src.repositories.market_data_repository import MarketDataRepository
from src.analytics.features import add_log_returns, add_sma
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

MODEL_PATH = max(Path("artifacts/models").glob("*.joblib"))
model = load_model(MODEL_PATH)

FEATURES = ["log_return", "sma_20", "sma_50"]


# =========================
# Sidebar controls
# =========================
st.sidebar.header("Configuration")

tickers = repo.list_tickers()
if not tickers:
    st.error("No tickers found. Run ingestion first.")
    st.stop()

ticker = st.sidebar.selectbox("Select NSE Ticker", tickers)

threshold = st.sidebar.slider(
    "Buy Probability Threshold",
    min_value=0.50,
    max_value=0.70,
    value=0.55,
    step=0.01,
)


# =========================
# Load & prepare data
# =========================
df = repo.load(ticker)
df = add_log_returns(df)
df = add_sma(df, 20)
df = add_sma(df, 50)
df = df.dropna()

if df.empty:
    st.error("Not enough data after feature engineering.")
    st.stop()


# =========================
# ML targets & predictions
# =========================
df["target"] = (df["log_return"].shift(-1) > 0).astype(int)
df = df.dropna()

proba = model.predict_proba(df[FEATURES])

signals = pd.Series(
    (proba > threshold).astype(int),
    index=df.index,
)

preds = signals.values
cls_metrics = classification_metrics(df["target"], preds)


# =========================
# Latest signal
# =========================
latest_proba = proba[-1]
latest_signal = int(latest_proba > threshold)

c1, c2, c3 = st.columns(3)

c1.metric("Latest Close Price", f"₹{df['Close'].iloc[-1]:,.2f}")
c2.metric("Probability Up", f"{latest_proba:.2%}")
c3.metric("Signal", "BUY" if latest_signal == 1 else "HOLD")


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
# Performance metrics
# =========================
st.subheader("Strategy Performance Metrics")

m1, m2, m3, m4, m5 = st.columns(5)

m1.metric("Total Return", f"{total_return(equity):.2%}")
m2.metric("Annualized Return", f"{annualized_return(returns):.2%}")
m3.metric("Sharpe Ratio", f"{sharpe_ratio(returns):.2f}")
m4.metric("Max Drawdown", f"{max_drawdown(equity):.2%}")
m5.metric("Win Rate", f"{win_rate(returns):.2%}")


# =========================
# Equity curve
# =========================
st.subheader("Strategy Equity Curve")

equity_fig = go.Figure()
equity_fig.add_trace(
    go.Scatter(
        x=bt["Date"],
        y=bt["equity_curve"],
        name="Strategy Equity",
    )
)
equity_fig.update_layout(height=400)
st.plotly_chart(equity_fig, use_container_width=True)


# =========================
# Validation metrics
# =========================
st.subheader("ML Validation Metrics")

v1, v2, v3, v4 = st.columns(4)

v1.metric("Accuracy", f"{cls_metrics['accuracy']:.2%}")
v2.metric("Precision", f"{cls_metrics['precision']:.2%}")
v3.metric("Recall", f"{cls_metrics['recall']:.2%}")
v4.metric("F1 Score", f"{cls_metrics['f1']:.2%}")


# =========================
# Footer
# =========================
st.caption("⚠️ Educational use only. Signals are generated from historical data.")
