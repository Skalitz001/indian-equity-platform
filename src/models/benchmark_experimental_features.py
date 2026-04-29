import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.analytics.experimental_features import (
    EXPERIMENTAL_FEATURE_COLUMNS,
    build_experimental_feature_frame,
)
from src.models.train_walkforward import BLEND_WEIGHTS
from src.repositories.market_data_repository import MarketDataRepository
from src.validation.metrics import classification_metrics_from_probabilities


ARTIFACTS_DIR = Path("artifacts/models")
OUTPUT_DIR = Path("artifacts/experimental_feature_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_meta(ticker: str) -> dict:
    meta_path = ARTIFACTS_DIR / f"{ticker.replace('.','_')}_meta.json"
    return json.loads(meta_path.read_text())


def classification_sort_key(metrics: dict) -> tuple:
    return (
        float(metrics["f1"]),
        float(metrics["precision"]),
        float(metrics["recall"]),
        float(metrics["accuracy"]),
        -abs(float(metrics.get("positive_rate", 0.0)) - 0.5),
    )


def choose_n_splits(df: pd.DataFrame) -> int:
    return max(3, min(5, len(df) // 200))


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = build_experimental_feature_frame(df, dropna=False).copy()

    next_log_return = frame["log_return"].shift(-1)
    frame["target"] = np.where(
        next_log_return.notna(),
        (next_log_return > 0).astype(int),
        np.nan,
    )

    return frame.dropna(subset=[*EXPERIMENTAL_FEATURE_COLUMNS, "target"]).reset_index(drop=True)


def build_logistic_builder(config: dict):
    def model_builder(config=config):
        return Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(
                C=config["C"],
                class_weight=config["class_weight"],
                max_iter=3000,
                solver="lbfgs",
                random_state=42,
            )),
        ])

    return model_builder


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

        probabilities.iloc[test_idx] = model.predict_proba(
            test_df[feature_columns]
        )[:, 1]

    return probabilities.dropna()


def evaluate_model(
    y_true: pd.Series,
    probabilities: pd.Series,
    label: str,
    config: dict | None = None,
) -> dict:
    metrics = classification_metrics_from_probabilities(
        y_true,
        probabilities,
        optimize_threshold=True,
    )

    return {
        "label": label,
        "config": config,
        **metrics,
        "support": int(metrics["support"]),
    }


def evaluate_ensemble(
    y_true: pd.Series,
    logistic_probabilities: pd.Series,
    rf_probabilities: pd.Series,
) -> dict:
    best = None
    common_index = logistic_probabilities.index.intersection(
        rf_probabilities.index
    )

    log_probs = logistic_probabilities.loc[common_index]
    rf_probs = rf_probabilities.loc[common_index]
    aligned_target = y_true.loc[common_index]

    for logistic_weight in BLEND_WEIGHTS:
        probabilities = (
            (logistic_weight * log_probs)
            + ((1 - logistic_weight) * rf_probs)
        )

        result = evaluate_model(
            aligned_target,
            probabilities,
            label=f"ensemble_{logistic_weight:.2f}",
            config={
                "logistic_weight": float(logistic_weight),
                "rf_weight": float(1 - logistic_weight),
            },
        )

        if best is None or classification_sort_key(result) > classification_sort_key(best):
            best = result

    return best


def best_baseline_snapshot(meta: dict) -> tuple[str, dict]:
    snapshots = {
        model_name: {
            "label": meta["oof_metrics"][model_name]["label"],
            **meta["oof_classification_metrics"][model_name],
        }
        for model_name in ("logistic", "rf", "ensemble")
    }

    best_model_name = max(
        snapshots,
        key=lambda model_name: classification_sort_key(snapshots[model_name]),
    )

    return best_model_name, snapshots[best_model_name]


def benchmark_ticker(ticker: str, repo: MarketDataRepository) -> dict | None:
    meta = load_meta(ticker)
    raw_df = repo.load(ticker)
    frame = build_training_frame(raw_df)

    if len(frame) < 250:
        return None

    logistic_probabilities = walk_forward_probabilities(
        frame,
        EXPERIMENTAL_FEATURE_COLUMNS,
        build_logistic_builder(meta["selected_configs"]["logistic"]),
    )
    rf_probabilities = walk_forward_probabilities(
        frame,
        EXPERIMENTAL_FEATURE_COLUMNS,
        build_rf_builder(meta["selected_configs"]["rf"]),
    )

    common_index = logistic_probabilities.index.intersection(
        rf_probabilities.index
    )
    target = frame.loc[common_index, "target"]

    logistic_result = evaluate_model(
        target,
        logistic_probabilities.loc[common_index],
        label="logistic",
        config=meta["selected_configs"]["logistic"],
    )
    rf_result = evaluate_model(
        target,
        rf_probabilities.loc[common_index],
        label="rf",
        config=meta["selected_configs"]["rf"],
    )
    ensemble_result = evaluate_ensemble(
        target,
        logistic_probabilities.loc[common_index],
        rf_probabilities.loc[common_index],
    )

    experimental_results = {
        "logistic": logistic_result,
        "rf": rf_result,
        "ensemble": ensemble_result,
    }
    best_experimental_model = max(
        experimental_results,
        key=lambda model_name: classification_sort_key(experimental_results[model_name]),
    )
    best_experimental = experimental_results[best_experimental_model]

    baseline_best_model, baseline_best = best_baseline_snapshot(meta)

    return {
        "ticker": ticker,
        "rows": int(len(frame)),
        "start_date": str(pd.Timestamp(frame["Date"].min()).date()),
        "end_date": str(pd.Timestamp(frame["Date"].max()).date()),
        "baseline_best_model": baseline_best_model,
        "baseline_best_label": baseline_best["label"],
        "baseline_precision": float(baseline_best["precision"]),
        "baseline_recall": float(baseline_best["recall"]),
        "baseline_f1": float(baseline_best["f1"]),
        "baseline_accuracy": float(baseline_best["accuracy"]),
        "baseline_threshold": float(baseline_best["threshold"]),
        "experimental_best_model": best_experimental_model,
        "experimental_best_label": best_experimental["label"],
        "experimental_precision": float(best_experimental["precision"]),
        "experimental_recall": float(best_experimental["recall"]),
        "experimental_f1": float(best_experimental["f1"]),
        "experimental_accuracy": float(best_experimental["accuracy"]),
        "experimental_threshold": float(best_experimental["threshold"]),
        "delta_precision": float(best_experimental["precision"] - baseline_best["precision"]),
        "delta_recall": float(best_experimental["recall"] - baseline_best["recall"]),
        "delta_f1": float(best_experimental["f1"] - baseline_best["f1"]),
        "delta_accuracy": float(best_experimental["accuracy"] - baseline_best["accuracy"]),
        "support": int(best_experimental["support"]),
        "strict_improvement": bool(
            best_experimental["precision"] > baseline_best["precision"]
            and best_experimental["recall"] > baseline_best["recall"]
            and best_experimental["f1"] > baseline_best["f1"]
        ),
        "logistic_precision": float(logistic_result["precision"]),
        "logistic_recall": float(logistic_result["recall"]),
        "logistic_f1": float(logistic_result["f1"]),
        "rf_precision": float(rf_result["precision"]),
        "rf_recall": float(rf_result["recall"]),
        "rf_f1": float(rf_result["f1"]),
        "ensemble_precision": float(ensemble_result["precision"]),
        "ensemble_recall": float(ensemble_result["recall"]),
        "ensemble_f1": float(ensemble_result["f1"]),
    }


def aggregate_summary(frame: pd.DataFrame) -> dict:
    return {
        "tickers": int(len(frame)),
        "improved_f1_count": int((frame["delta_f1"] > 0).sum()),
        "improved_precision_count": int((frame["delta_precision"] > 0).sum()),
        "improved_recall_count": int((frame["delta_recall"] > 0).sum()),
        "strict_improvement_count": int(frame["strict_improvement"].sum()),
        "avg_baseline_precision": float(frame["baseline_precision"].mean()),
        "avg_experimental_precision": float(frame["experimental_precision"].mean()),
        "avg_baseline_recall": float(frame["baseline_recall"].mean()),
        "avg_experimental_recall": float(frame["experimental_recall"].mean()),
        "avg_baseline_f1": float(frame["baseline_f1"].mean()),
        "avg_experimental_f1": float(frame["experimental_f1"].mean()),
        "avg_delta_precision": float(frame["delta_precision"].mean()),
        "avg_delta_recall": float(frame["delta_recall"].mean()),
        "avg_delta_f1": float(frame["delta_f1"].mean()),
        "avg_delta_accuracy": float(frame["delta_accuracy"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark an experimental market-context feature set.",
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
        row = benchmark_ticker(ticker, repo)
        if row is not None:
            rows.append(row)

    frame = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    summary = aggregate_summary(frame)

    frame.to_csv(OUTPUT_DIR / "experimental_features_main.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
