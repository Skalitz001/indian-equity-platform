import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd

from src.repositories.market_data_repository import MarketDataRepository
from src.analytics.features import add_log_returns, add_sma
from src.models.persistence import load_model
from src.backtesting.engine import BacktestEngine

FEATURES = ["log_return", "sma_20", "sma_50"]
MODEL_PATH = max(Path("artifacts/models").glob("*.joblib"))

st.set_page_config(
    page_title="Indian Equity ML Signals",
    layout="wide",
)

st.title("📈 Indian Equity ML Signal Dashboard")
st.caption("End-to-end analytics + ML platform (NSE stocks)")

# Load shared resources
repo = MarketDataRepository()
model = load_model(MODEL_PATH)
engine = BacktestEngine(transaction_cost=0.001)

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.selectbox(
    "Select NSE Ticker",
    options=repo.list_tickers(),
)
threshold = st.sidebar.slider(
    "Buy Probability Threshold",
    min_value=0.50,
    max_value=0.70,
    value=0.55,
    step=0.01,
)

# Load data
df = repo.load(ticker)
df = add_log_returns(df)
df = add_sma(df, 20)
df = add_sma(df, 50)
df = df.dropna()

# Latest prediction
latest_proba = model.predict_proba(df[FEATURES])[-1]
latest_signal = int(latest_proba > threshold)

# Layout
col1, col2, col3 = st.columns(3)

col1.metric(
    "Latest Close Price",
    f"₹{df['Close'].iloc[-1]:,.2f}",
)

col2.metric(
    "Probability Up",
    f"{latest_proba:.2%}",
)

col3.metric(
    "Signal",
    "BUY" if latest_signal == 1 else "HOLD",
)

# Price chart
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
price_fig.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(price_fig, use_container_width=True)

# Backtest
proba = model.predict_proba(df[FEATURES])
signals = pd.Series(
    (proba > threshold).astype(int),
    index=df.index,
)

bt = engine.run(df, signals)

# Equity curve
st.subheader("Strategy Backtest")

equity_fig = go.Figure()
equity_fig.add_trace(
    go.Scatter(
        x=bt["Date"],
        y=bt["equity_curve"],
        name="Strategy Equity",
    )
)
equity_fig.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=30, b=20),
)

st.plotly_chart(equity_fig, use_container_width=True)

# Footer
st.caption(
    "⚠️ Educational use only. Signals are generated from historical data."
)
