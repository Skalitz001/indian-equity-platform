# Indian Equity ML Signal Platform

This repository contains a research platform for Indian equity direction prediction, walk-forward model selection, strategy backtesting, and Streamlit-based signal review.

The project supports two production-facing pipelines:

- `Main Pipeline`: stock-only features used to predict the next trading session's close-to-close direction.
- `GIFT-Aware Pipeline`: stock history plus pre-open GIFT Nifty features used to predict the same Indian cash-market session's open-to-close direction.

The code is intended for research and education. It is not investment advice and should not be used for live trading without independent validation, execution modelling, and risk controls.

## Core Idea

GIFT Nifty trades on Singapore market hours and can provide information before the Indian cash market opens. The GIFT-aware pipeline is therefore framed as a pre-open signal for the Indian session:

- Stock-derived features are shifted by one session so they are known before the Indian open.
- Same-date GIFT rows are treated as completed Singapore-time pre-open observations, available before the NSE cash session.
- The target is same-day Indian equity `open -> close` direction, not next-day `close -> close` direction.
- The model records `gift_source_age_days` so stale GIFT observations are visible to the classifier.

This timing choice is deliberate. Injecting GIFT directly into the main next-day close-to-close target would mix incompatible information timing and weaken the interpretation of the model.

## Pipeline Summary

| Pipeline | Inputs | Prediction Target | Signal Policy | Output Artifacts |
| --- | --- | --- | --- | --- |
| Main | Stock OHLCV | Next-session close-to-close up/down | Long/flat state machine with entry and exit thresholds | `artifacts/models/` |
| GIFT-Aware | Stock OHLCV and GIFT Nifty OHLC | Same-session open-to-close up/down | Independent intraday participation threshold | `artifacts/gift_models/` |
| Comparison | Main, aligned baseline, and GIFT outputs | Relative strategy and classifier performance | Report generation | `artifacts/gift_models/comparison/` |

## Model Workflow

Both pipelines follow the same high-level lifecycle:

1. Load local OHLCV data.
2. Build time-safe feature frames.
3. Train logistic regression and random forest candidates.
4. Evaluate out-of-fold probabilities with walk-forward validation.
5. Search probability thresholds using backtest quality, not classifier accuracy alone.
6. Fit final selected models and weighted ensembles.
7. Save model, metadata, metrics, signal policy, holdout metrics, and prediction-store artifacts.
8. Display current signals and historical diagnostics in the dashboard.

Model ranking prioritizes:

1. Sharpe ratio.
2. Total return.
3. Maximum drawdown.
4. Win rate.
5. Trade activity and tie-breakers.

## Repository Layout

```text
dashboard/
  app.py                  Streamlit entrypoint
  controller.py           Dashboard state assembly and orchestration
  views.py                Dashboard tabs, charts, tables, and metrics
  outlook.py              Three-session outlook logic
  pipeline.py             Dashboard data/model loading helpers
  theme.py                Dashboard CSS and chart styling
  figures.py              Reusable Plotly comparison figures
  probability_display.py  Prediction label and probability formatting

src/
  analytics/              Feature engineering and performance utilities
  backtesting/            Main close-to-close backtest engine
  gift_nifty/             GIFT ingestion, alignment, features, training, prediction
  models/                 Main training scripts and benchmark experiments
  repositories/           Local market data access
  strategies/             Signal policy utilities
  validation/             Metrics, walk-forward validation, prediction-store helpers

data/
  raw/                    Local stock and index OHLCV CSV files
  external/               Normalized external inputs such as GIFT Nifty

artifacts/
  models/                 Main pipeline model artifacts
  gift_models/            GIFT-aware model artifacts
```

## Data Requirements

The current release path expects local CSV inputs:

- Stock and index files in `data/raw/`, for example `TCS.NS.csv` and `^NSEI.csv`.
- GIFT Nifty history in `data/external/gift_nifty.csv`.

The GIFT input must represent data that is available before the Indian cash market opens. If the source instead represents after-close values, the GIFT-aware alignment must be changed before training.

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the test suite:

```bash
python3 -m pytest
```

## Training Artifacts

Train the dashboard-facing artifacts in this order.

Train the main stock-only pipeline:

```bash
python3 -m src.models.train_walkforward
```

Train all GIFT-aware models:

```bash
python3 -m src.gift_nifty.train_walkforward
```

Rebuild the GIFT comparison report:

```bash
python3 -m src.gift_nifty.compare_pipelines
```

Optional: refresh the normalized GIFT Nifty CSV before training:

```bash
python3 -m src.gift_nifty.ingest_dhan
```

Optional: train one GIFT-aware ticker:

```bash
python3 -m src.gift_nifty.train_walkforward --ticker TCS.NS
```

Optional: inspect the latest GIFT-aware prediction for one ticker:

```bash
python3 -m src.gift_nifty.predict --ticker TCS.NS --model-name ensemble
```

## Dashboard

Run the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard provides:

- Pipeline selection between main and GIFT-aware models.
- Ticker and model selection.
- Latest classifier call with `P(Up)` or `P(Down)` shown according to the predicted direction.
- Red markers for down predictions and green markers for up predictions.
- Probability and price curves.
- Three-session outlook.
- Strategy returns, drawdown, and benchmark comparisons.
- Classifier metrics and recent model-ready rows.
- Pipeline comparison charts for the GIFT-aware workflow.

## Validation Design

The project avoids random train/test splits for model selection. It uses chronological walk-forward validation so that each validation fold is evaluated after its training history.

Recent training code also writes holdout metrics and prediction-store files so future changes can be compared against saved prediction streams rather than only aggregate backtest summaries.

Key validation constraints:

- Stock features used by the GIFT pipeline are lagged by one session.
- GIFT rows are merged backward by date and never from the future.
- GIFT same-date rows are accepted only under the pre-open availability assumption.
- Main and GIFT-native metrics are not directly comparable because they predict different targets on different execution schedules.
- The valid GIFT comparison is against an aligned stock-only intraday baseline.

## Important Files

- `src/models/train_walkforward.py`: main walk-forward trainer.
- `src/gift_nifty/train_walkforward.py`: GIFT-aware walk-forward trainer.
- `src/gift_nifty/dataset.py`: stock and GIFT alignment rules.
- `src/gift_nifty/features.py`: GIFT feature engineering.
- `src/gift_nifty/compare_pipelines.py`: aligned comparison reporting.
- `dashboard/controller.py`: dashboard state assembly.
- `dashboard/views.py`: dashboard rendering.
- `src/validation/prediction_store.py`: saved prediction stream helpers.

## Limitations

- Results depend on the quality, completeness, and timestamp semantics of local CSV data.
- Backtests use simplified transaction-cost and execution assumptions.
- Liquidity, slippage, corporate actions, taxes, borrow constraints, and operational risk are not fully modelled.
- Model performance can decay and must be monitored out of sample.
- A strong backtest does not imply future profitability.

## Development Checklist

Before pushing changes:

```bash
python3 -m compileall -q dashboard src tests
python3 -m pytest
git diff --check
```

Review generated artifacts before committing them. Large model files and regenerated reports should be committed only when they are intentionally part of the project snapshot.
