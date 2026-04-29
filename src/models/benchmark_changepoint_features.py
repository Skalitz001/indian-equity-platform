import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from src.analytics.changepoint_features import (
    CHANGEPOINT_FEATURE_COLUMNS,
    PAPER_CHANGEPOINT_FEATURE_COLUMNS,
    add_changepoint_features,
)
from src.analytics.experimental_features import (
    MARKET_CONTEXT_FEATURE_COLUMNS,
    MARKET_CONTEXT_FEATURE_GROUPS,
    build_experimental_feature_frame,
)
from src.analytics.features import FEATURE_COLUMNS
from src.backtesting.engine import BacktestEngine
from src.models.probabilities import predict_up_probability
from src.models.train_walkforward import model_sort_key, select_threshold
from src.repositories.market_data_repository import MarketDataRepository
from src.validation.metrics import classification_metrics_from_probabilities


ARTIFACTS_DIR = Path("artifacts/models")
OUTPUT_DIR = Path("artifacts/changepoint_feature_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSTANTIAL_SHARPE_DELTA = 0.003
SUBSTANTIAL_IMPROVED_RATE = 0.60

CHANGEPOINT_FEATURE_GROUPS = {
    "changepoint": CHANGEPOINT_FEATURE_COLUMNS,
    "paper_changepoint": [
        *MARKET_CONTEXT_FEATURE_COLUMNS,
        *PAPER_CHANGEPOINT_FEATURE_COLUMNS,
    ],
    "activity_risk_changepoint": [
        *MARKET_CONTEXT_FEATURE_GROUPS["activity_risk"],
        *CHANGEPOINT_FEATURE_COLUMNS,
    ],
    "context_changepoint": [
        *MARKET_CONTEXT_FEATURE_COLUMNS,
        *CHANGEPOINT_FEATURE_COLUMNS,
    ],
}


def load_meta(ticker: str) -> dict:
    meta_path = ARTIFACTS_DIR / f"{ticker.replace('.','_')}_meta.json"
    return json.loads(meta_path.read_text())


def choose_n_splits(df: pd.DataFrame) -> int:
    return max(3, min(5, len(df) // 200))


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = build_experimental_feature_frame(df, dropna=False).copy()
    frame = add_changepoint_features(frame)

    next_log_return = frame["log_return"].shift(-1)
    frame["target"] = np.where(
        next_log_return.notna(),
        (next_log_return > 0).astype(int),
        np.nan,
    )

    required_columns = [
        *FEATURE_COLUMNS,
        *MARKET_CONTEXT_FEATURE_COLUMNS,
        *CHANGEPOINT_FEATURE_COLUMNS,
        "target",
    ]

    return frame.dropna(subset=required_columns).reset_index(drop=True)


def build_rf_builder(config: dict):
    def model_builder(config=config):
        return RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    return model_builder


def walk_forward_probabilities(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
) -> pd.Series:
    probabilities = pd.Series(index=df.index, dtype=float)
    splitter = TimeSeriesSplit(n_splits=choose_n_splits(df))

    for train_idx, test_idx in splitter.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = model_builder()
        model.fit(
            train_df[feature_columns],
            train_df["target"],
        )

        probabilities.iloc[test_idx] = predict_up_probability(
            model,
            test_df[feature_columns],
        )

    return probabilities.dropna()


def evaluate_feature_set(
    df: pd.DataFrame,
    feature_columns: list[str],
    config: dict,
    label: str,
    engine: BacktestEngine,
) -> dict:
    probabilities = walk_forward_probabilities(
        df,
        feature_columns,
        build_rf_builder(config),
    )

    classification = classification_metrics_from_probabilities(
        df.loc[probabilities.index, "target"],
        probabilities,
        optimize_threshold=True,
    )
    trading = select_threshold(
        df,
        probabilities,
        engine,
    )

    return {
        "label": label,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "classification": classification,
        "trading": trading,
    }


def classification_sort_key(result: dict) -> tuple:
    metrics = result["classification"]
    return (
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["accuracy"]),
        -abs(float(metrics["positive_rate"]) - 0.5),
    )


def benchmark_ticker(ticker: str, repo: MarketDataRepository) -> list[dict]:
    meta = load_meta(ticker)
    raw_df = repo.load(ticker)
    frame = build_training_frame(raw_df)

    if len(frame) < 250:
        return []

    engine = BacktestEngine(transaction_cost=0.001)
    rf_config = meta["selected_configs"]["rf"]

    baseline = evaluate_feature_set(
        frame,
        FEATURE_COLUMNS,
        rf_config,
        label="baseline_rf",
        engine=engine,
    )

    rows = []

    for group_name, group_columns in CHANGEPOINT_FEATURE_GROUPS.items():
        candidate_columns = [*FEATURE_COLUMNS, *group_columns]
        candidate = evaluate_feature_set(
            frame,
            candidate_columns,
            rf_config,
            label=group_name,
            engine=engine,
        )

        baseline_cls = baseline["classification"]
        candidate_cls = candidate["classification"]
        baseline_trading = baseline["trading"]
        candidate_trading = candidate["trading"]

        rows.append({
            "ticker": ticker,
            "rows": int(len(frame)),
            "start_date": str(pd.Timestamp(frame["Date"].min()).date()),
            "end_date": str(pd.Timestamp(frame["Date"].max()).date()),
            "group_name": group_name,
            "feature_count": int(candidate["feature_count"]),
            "baseline_precision": float(baseline_cls["precision"]),
            "baseline_recall": float(baseline_cls["recall"]),
            "baseline_f1": float(baseline_cls["f1"]),
            "baseline_accuracy": float(baseline_cls["accuracy"]),
            "baseline_sharpe": float(baseline_trading["sharpe"]) if baseline_trading else np.nan,
            "baseline_total_return": float(baseline_trading["total_return"]) if baseline_trading else np.nan,
            "candidate_precision": float(candidate_cls["precision"]),
            "candidate_recall": float(candidate_cls["recall"]),
            "candidate_f1": float(candidate_cls["f1"]),
            "candidate_accuracy": float(candidate_cls["accuracy"]),
            "candidate_sharpe": float(candidate_trading["sharpe"]) if candidate_trading else np.nan,
            "candidate_total_return": float(candidate_trading["total_return"]) if candidate_trading else np.nan,
            "delta_precision": float(candidate_cls["precision"] - baseline_cls["precision"]),
            "delta_recall": float(candidate_cls["recall"] - baseline_cls["recall"]),
            "delta_f1": float(candidate_cls["f1"] - baseline_cls["f1"]),
            "delta_accuracy": float(candidate_cls["accuracy"] - baseline_cls["accuracy"]),
            "delta_sharpe": float(candidate_trading["sharpe"] - baseline_trading["sharpe"]) if baseline_trading and candidate_trading else np.nan,
            "delta_total_return": float(candidate_trading["total_return"] - baseline_trading["total_return"]) if baseline_trading and candidate_trading else np.nan,
            "strict_classification_improvement": bool(
                candidate_cls["precision"] > baseline_cls["precision"]
                and candidate_cls["recall"] > baseline_cls["recall"]
                and candidate_cls["f1"] > baseline_cls["f1"]
            ),
            "classification_improved": bool(
                classification_sort_key(candidate) > classification_sort_key(baseline)
            ),
            "trading_improved": bool(
                candidate_trading is not None
                and baseline_trading is not None
                and model_sort_key(candidate_trading) > model_sort_key(baseline_trading)
            ),
        })

    return rows


def aggregate_summary(frame: pd.DataFrame) -> dict:
    summary = {}

    for group_name, group_frame in frame.groupby("group_name"):
        tickers = int(len(group_frame))
        trading_improved_count = int(group_frame["trading_improved"].sum())
        avg_delta_sharpe = float(group_frame["delta_sharpe"].mean())
        avg_delta_total_return = float(group_frame["delta_total_return"].mean())
        improved_rate = trading_improved_count / tickers if tickers else 0.0

        summary[group_name] = {
            "tickers": tickers,
            "classification_improved_count": int(
                group_frame["classification_improved"].sum()
            ),
            "strict_classification_improvement_count": int(
                group_frame["strict_classification_improvement"].sum()
            ),
            "trading_improved_count": trading_improved_count,
            "trading_improved_rate": float(improved_rate),
            "avg_delta_precision": float(group_frame["delta_precision"].mean()),
            "avg_delta_recall": float(group_frame["delta_recall"].mean()),
            "avg_delta_f1": float(group_frame["delta_f1"].mean()),
            "avg_delta_accuracy": float(group_frame["delta_accuracy"].mean()),
            "avg_delta_sharpe": avg_delta_sharpe,
            "avg_delta_total_return": avg_delta_total_return,
            "substantial_improvement": bool(
                avg_delta_sharpe >= SUBSTANTIAL_SHARPE_DELTA
                and improved_rate >= SUBSTANTIAL_IMPROVED_RATE
                and avg_delta_total_return > 0.0
            ),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prototype changepoint feature groups against baseline RF features.",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Ticker to benchmark. Pass multiple times to limit the run.",
    )
    args = parser.parse_args()

    repo = MarketDataRepository()
    tickers = args.tickers or sorted(
        path.stem.replace("_meta", "").replace("_", ".")
        for path in ARTIFACTS_DIR.glob("*_meta.json")
    )

    rows = []

    for ticker in tickers:
        rows.extend(benchmark_ticker(ticker, repo))

    frame = pd.DataFrame(rows).sort_values(["group_name", "ticker"]).reset_index(drop=True)
    summary = aggregate_summary(frame)

    frame.to_csv(OUTPUT_DIR / "changepoint_feature_results.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
