 Indian Equity ML Trading Platform

An end-to-end machine learning research platform for algorithmic trading in Indian equities.
The project integrates data engineering, feature engineering, machine learning, walk-forward validation, backtesting, and interactive visualization into a modular software system.

The objective is to investigate whether machine learning signals can improve risk-adjusted returns compared to a traditional Buy-and-Hold strategy.

Key Features

End-to-end ML trading research pipeline

Automated market data ingestion

Advanced feature engineering for financial time series

Walk-forward training to prevent look-ahead bias

Sharpe-ratio-optimized signal thresholds

Integrated vectorized backtesting engine

Streamlit interactive dashboard

Buy-and-Hold benchmark comparison

Modular software-engineering architecture

🏗 System Architecture

The platform is designed as a layered architecture separating responsibilities across modules.

Market Data
     │
     ▼
Data Ingestion
     │
     ▼
Feature Engineering
     │
     ▼
Machine Learning Model
     │
     ▼
Signal Generation
     │
     ▼
Backtesting Engine
     │
     ▼
Interactive Dashboard

This structure ensures:

modular experimentation

reproducible pipelines

clean separation of data, models, and UI

 Project Structure
indian-equity-platform
│
├── configs
│   └── data.yaml
│
├── data
│   └── raw
│
├── artifacts
│   └── models
│
├── src
│   ├── ingestion
│   │   └── nse_downloader.py
│   │
│   ├── repositories
│   │   └── market_data_repository.py
│   │
│   ├── analytics
│   │   ├── features.py
│   │   └── performance.py
│   │
│   ├── models
│   │   ├── logistic_model.py
│   │   ├── persistence.py
│   │   └── train_walkforward.py
│   │
│   ├── backtesting
│   │   └── engine.py
│   │
│   ├── validation
│   │   └── metrics.py
│   │
│   └── utils
│
└── dashboard
    └── app.py
 Stock Universe

The project evaluates 10 large-cap NSE equities across two sectors.

Information Technology

TCS

Infosys

HCL Technologies

Wipro

Tech Mahindra

Banking & Financial Services

HDFC Bank

ICICI Bank

State Bank of India

Axis Bank

Kotak Mahindra Bank

Using two sectors allows the model to be tested across different market regimes and economic drivers.

 Data Pipeline

Market data is downloaded using Yahoo Finance (yfinance).

Example storage format:

data/raw/TCS.NS.csv
data/raw/INFY.NS.csv
data/raw/HDFCBANK.NS.csv

Each dataset contains:

Open

High

Low

Close

Volume

Date

 Feature Engineering

The platform extracts several categories of financial features.

Momentum Indicators

Log Returns

10-day Momentum

MACD Histogram

Trend Indicators

20-day SMA

50-day SMA

SMA Difference

SMA Ratio

Volatility Features

Rolling Volatility

Average True Range (ATR)

Market Regime Features

Rolling Skewness

Rolling Kurtosis

Liquidity Indicators

Volume Change

Volume Moving Average

These features allow the model to capture:

price trends

volatility regimes

momentum bursts

liquidity shifts

 Machine Learning Model

The system uses Logistic Regression as the baseline classifier.

Target variable:

target = 1 if next-day return > 0
target = 0 otherwise

Model pipeline:

Feature Scaling
      ↓
Logistic Regression
      ↓
Probability Calibration

Predictions produce a probability:

P(price_up)
 Walk-Forward Training

The model uses time-series cross-validation to prevent data leakage.

Train Window → Validate → Expand Window → Repeat

During training:

Regularization strength is tuned

Probability calibration is applied

Trading threshold is optimized using Sharpe Ratio

 Backtesting Engine

The project includes a lightweight vectorized backtesting engine.

Strategy assumptions:

Long-only

Trades executed next day

Transaction costs included

Daily rebalancing

Metrics calculated:

Total Return

Annualized Return

Sharpe Ratio

Maximum Drawdown

Win Rate

Performance is compared against a Buy-and-Hold benchmark.

Interactive Dashboard

The Streamlit dashboard allows interactive exploration of model signals.

Features include:

stock selection

probability-based signals

price chart visualization

equity curve plotting

ML performance metrics

strategy vs benchmark comparison

Run dashboard:

streamlit run dashboard/app.py
Training the Model

Train models with walk-forward validation:

python3 -m src.models.train_walkforward

Models are saved to:

artifacts/models/

Each artifact stores:

trained model

optimized threshold

hyperparameters

cross-validation Sharpe

▶ Running the Full Pipeline
1️⃣ Download data
python3 -m src.ingestion.nse_downloader
2️⃣ Train models
python3 -m src.models.train_walkforward
3️⃣ Launch dashboard
streamlit run dashboard/app.py
 Evaluation Metrics

Two evaluation layers are used.

Machine Learning Metrics

Accuracy

Precision

Recall

F1 Score

Trading Metrics

Sharpe Ratio

Maximum Drawdown

Total Return

Win Rate

Trading performance is always compared against Buy-and-Hold.

 Technologies Used

Python

Pandas

NumPy

Scikit-learn

Streamlit

Plotly

yfinance

⚠ Disclaimer

This project is intended for academic and research purposes only.

The trading signals generated by the model should not be used for live financial trading without further validation and risk management.