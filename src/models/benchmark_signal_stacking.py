import argparse
import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from src.backtesting.engine import BacktestEngine
from src.gift_nifty.backtesting import IntradayBacktestEngine
from src.gift_nifty.constants import DEFAULT_GIFT_DATA_PATH
from src.gift_nifty.train_walkforward import (
    build_training_frame,
    choose_n_splits as choose_gift_n_splits,
    model_sort_key as gift_model_sort_key,
    oof_classification_metrics as gift_oof_classification_metrics,
    select_threshold as select_gift_threshold,
    walk_forward_probabilities as gift_walk_forward_probabilities,
)
from src.models.probabilities import predict_up_probability
from src.models.train_walkforward import (
    build_features as build_main_features,
    choose_n_splits as choose_main_n_splits,
    model_sort_key as main_model_sort_key,
    oof_classification_metrics as main_oof_classification_metrics,
    select_threshold as select_main_threshold,
    walk_forward_probabilities as main_walk_forward_probabilities,
)
from src.repositories.market_data_repository import MarketDataRepository
from src.strategies.signal_policy import generate_probability_signals


OUTPUT_DIR = Path("artifacts/stacked_signal_boosting")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAIN_ARTIFACTS = Path("artifacts/models")
GIFT_ARTIFACTS = Path("artifacts/gift_models")

STACK_FEATURE_COLUMNS = [
    "logistic_prob",
    "rf_prob",
    "ensemble_prob",
    "logistic_signal",
    "rf_signal",
    "ensemble_signal",
    "prob_gap",
    "ensemble_confidence",
    "signal_agreement",
]

XGBOOST_GRID = (
    {
        "n_estimators": 60,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 1.0,
    },
    {
        "n_estimators": 120,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2.0,
    },
)

LIGHTGBM_GRID = (
    {
        "n_estimators": 60,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    {
        "n_estimators": 120,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
)


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text())


def meta_model_snapshot(meta: dict, model_name: str) -> dict:
    result = meta["oof_metrics"][model_name]
    cls = meta["oof_classification_metrics"][model_name]

    return {
        "label": result["label"],
        "threshold": float(result["threshold"]),
        "entry_threshold": float(result["entry_threshold"]),
        "exit_threshold": float(result["exit_threshold"]),
        "sharpe": float(result["sharpe"]),
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "win_rate": float(result["win_rate"]),
        "active_days": int(result["active_days"]),
        "entries": int(result["entries"]),
        "accuracy": float(cls["accuracy"]),
        "precision": float(cls["precision"]),
        "recall": float(cls["recall"]),
        "f1": float(cls["f1"]),
        "positive_rate": float(cls["positive_rate"]),
        "support": int(cls["support"]),
    }


def result_snapshot(result: dict) -> dict:
    cls = result["classification_metrics"]

    return {
        "label": result["label"],
        "threshold": float(result["threshold"]),
        "entry_threshold": float(result["entry_threshold"]),
        "exit_threshold": float(result["exit_threshold"]),
        "sharpe": float(result["sharpe"]),
        "total_return": float(result["total_return"]),
        "max_drawdown": float(result["max_drawdown"]),
        "win_rate": float(result["win_rate"]),
        "active_days": int(result["active_days"]),
        "entries": int(result["entries"]),
        "accuracy": float(cls["accuracy"]),
        "precision": float(cls["precision"]),
        "recall": float(cls["recall"]),
        "f1": float(cls["f1"]),
        "positive_rate": float(cls["positive_rate"]),
        "support": int(cls["support"]),
    }


def build_main_logistic_builder(config: dict):
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


def build_gift_logistic_builder(config: dict):
    return build_main_logistic_builder(config)


def build_main_rf_builder(config: dict):
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


def build_gift_rf_builder(config: dict):
    def model_builder(config=config):
        return RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            random_state=42,
        )

    return model_builder


def build_xgboost_candidates():
    candidates = []

    for config in XGBOOST_GRID:
        label = (
            f"xgb_ne{config['n_estimators']}_"
            f"md{config['max_depth']}_"
            f"lr{config['learning_rate']}"
        )

        def model_builder(config=config):
            return XGBClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                min_child_weight=config["min_child_weight"],
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
            )

        candidates.append({
            "label": label,
            "config": config,
            "builder": model_builder,
        })

    return candidates


def build_lightgbm_candidates():
    candidates = []

    for config in LIGHTGBM_GRID:
        label = (
            f"lgbm_ne{config['n_estimators']}_"
            f"nl{config['num_leaves']}_"
            f"lr{config['learning_rate']}"
        )

        def model_builder(config=config):
            return LGBMClassifier(
                n_estimators=config["n_estimators"],
                num_leaves=config["num_leaves"],
                learning_rate=config["learning_rate"],
                min_child_samples=config["min_child_samples"],
                subsample=config["subsample"],
                colsample_bytree=config["colsample_bytree"],
                objective="binary",
                random_state=42,
                n_jobs=1,
                verbosity=-1,
            )

        candidates.append({
            "label": label,
            "config": config,
            "builder": model_builder,
        })

    return candidates


def build_signal_stack_frame(
    frame: pd.DataFrame,
    meta: dict,
    logistic_probabilities: pd.Series,
    rf_probabilities: pd.Series,
) -> pd.DataFrame:
    common_index = logistic_probabilities.index.intersection(
        rf_probabilities.index
    )

    stack_frame = frame.loc[common_index].copy()

    stack_frame["logistic_prob"] = logistic_probabilities.loc[common_index]
    stack_frame["rf_prob"] = rf_probabilities.loc[common_index]

    weights = meta.get("blend_weights", {
        "logistic": 0.5,
        "rf": 0.5,
    })
    stack_frame["ensemble_prob"] = (
        float(weights["logistic"]) * stack_frame["logistic_prob"]
        + float(weights["rf"]) * stack_frame["rf_prob"]
    )

    thresholds = meta.get("thresholds", {})
    signal_policies = meta.get("signal_policies", {})

    for model_name in ("logistic", "rf", "ensemble"):
        stack_frame[f"{model_name}_signal"] = generate_probability_signals(
            stack_frame[f"{model_name}_prob"],
            signal_policies.get(
                model_name,
                thresholds.get(model_name, meta.get("opt_threshold", 0.55)),
            ),
        ).astype(float)

    stack_frame["prob_gap"] = (
        stack_frame["logistic_prob"] - stack_frame["rf_prob"]
    )
    stack_frame["ensemble_confidence"] = (
        stack_frame["ensemble_prob"] - 0.5
    ).abs()
    stack_frame["signal_agreement"] = (
        (stack_frame["logistic_signal"] == stack_frame["rf_signal"])
        & (stack_frame["rf_signal"] == stack_frame["ensemble_signal"])
    ).astype(float)

    return stack_frame


def walk_forward_meta_probabilities(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
    choose_n_splits_fn,
) -> pd.Series:
    probabilities = pd.Series(index=df.index, dtype=float)
    splitter = TimeSeriesSplit(n_splits=choose_n_splits_fn(df))

    for train_idx, test_idx in splitter.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        if train_df["target"].nunique() < 2:
            probabilities.iloc[test_idx] = float(train_df["target"].mean())
            continue

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


def evaluate_meta_candidates(
    df: pd.DataFrame,
    feature_columns: list[str],
    candidates: list[dict],
    choose_n_splits_fn,
    select_threshold_fn,
    oof_classification_metrics_fn,
    model_sort_key_fn,
    engine,
):
    best = None

    for candidate in candidates:
        probabilities = walk_forward_meta_probabilities(
            df,
            feature_columns,
            candidate["builder"],
            choose_n_splits_fn,
        )

        if probabilities.empty:
            continue

        metrics = select_threshold_fn(
            df,
            probabilities,
            engine,
        )

        if metrics is None:
            continue

        result = {
            **metrics,
            "label": candidate["label"],
            "config": candidate["config"],
            "classification_metrics": oof_classification_metrics_fn(
                df,
                probabilities,
            ),
            "probabilities": probabilities,
        }

        if best is None or model_sort_key_fn(result) > model_sort_key_fn(best):
            best = result

    return best


def choose_best_stacked_result(best_xgboost, best_lightgbm, model_sort_key_fn):
    candidates = {
        "xgboost": best_xgboost,
        "lightgbm": best_lightgbm,
    }
    available = {
        name: result
        for name, result in candidates.items()
        if result is not None
    }

    if not available:
        return None, None

    best_name = max(
        available,
        key=lambda model_name: model_sort_key_fn(available[model_name]),
    )

    return best_name, available[best_name]


def comparison_row(
    pipeline: str,
    ticker: str,
    frame: pd.DataFrame,
    current_model_name: str,
    current_snapshot: dict,
    ensemble_snapshot: dict,
    best_xgboost: dict | None,
    best_lightgbm: dict | None,
    model_sort_key_fn,
) -> dict:
    xgboost_snapshot = result_snapshot(best_xgboost) if best_xgboost else None
    lightgbm_snapshot = result_snapshot(best_lightgbm) if best_lightgbm else None
    best_stack_name, best_stack_result = choose_best_stacked_result(
        best_xgboost,
        best_lightgbm,
        model_sort_key_fn,
    )
    best_stack_snapshot = (
        result_snapshot(best_stack_result) if best_stack_result else None
    )

    return {
        "pipeline": pipeline,
        "ticker": ticker,
        "rows": int(len(frame)),
        "start_date": str(pd.Timestamp(frame["Date"].min()).date()),
        "end_date": str(pd.Timestamp(frame["Date"].max()).date()),
        "current_model": current_model_name,
        "current_label": current_snapshot["label"],
        "current_sharpe": current_snapshot["sharpe"],
        "current_total_return": current_snapshot["total_return"],
        "current_accuracy": current_snapshot["accuracy"],
        "current_f1": current_snapshot["f1"],
        "ensemble_label": ensemble_snapshot["label"],
        "ensemble_sharpe": ensemble_snapshot["sharpe"],
        "ensemble_total_return": ensemble_snapshot["total_return"],
        "ensemble_accuracy": ensemble_snapshot["accuracy"],
        "ensemble_f1": ensemble_snapshot["f1"],
        "xgboost_label": xgboost_snapshot["label"] if xgboost_snapshot else None,
        "xgboost_sharpe": xgboost_snapshot["sharpe"] if xgboost_snapshot else None,
        "xgboost_total_return": (
            xgboost_snapshot["total_return"] if xgboost_snapshot else None
        ),
        "xgboost_accuracy": (
            xgboost_snapshot["accuracy"] if xgboost_snapshot else None
        ),
        "xgboost_f1": xgboost_snapshot["f1"] if xgboost_snapshot else None,
        "lightgbm_label": lightgbm_snapshot["label"] if lightgbm_snapshot else None,
        "lightgbm_sharpe": (
            lightgbm_snapshot["sharpe"] if lightgbm_snapshot else None
        ),
        "lightgbm_total_return": (
            lightgbm_snapshot["total_return"] if lightgbm_snapshot else None
        ),
        "lightgbm_accuracy": (
            lightgbm_snapshot["accuracy"] if lightgbm_snapshot else None
        ),
        "lightgbm_f1": lightgbm_snapshot["f1"] if lightgbm_snapshot else None,
        "best_stack_model": best_stack_name,
        "best_stack_label": (
            best_stack_snapshot["label"] if best_stack_snapshot else None
        ),
        "best_stack_sharpe": (
            best_stack_snapshot["sharpe"] if best_stack_snapshot else None
        ),
        "best_stack_total_return": (
            best_stack_snapshot["total_return"] if best_stack_snapshot else None
        ),
        "best_stack_accuracy": (
            best_stack_snapshot["accuracy"] if best_stack_snapshot else None
        ),
        "best_stack_f1": best_stack_snapshot["f1"] if best_stack_snapshot else None,
        "best_stack_minus_current_sharpe": (
            best_stack_snapshot["sharpe"] - current_snapshot["sharpe"]
            if best_stack_snapshot else None
        ),
        "best_stack_minus_current_total_return": (
            best_stack_snapshot["total_return"] - current_snapshot["total_return"]
            if best_stack_snapshot else None
        ),
        "best_stack_minus_current_accuracy": (
            best_stack_snapshot["accuracy"] - current_snapshot["accuracy"]
            if best_stack_snapshot else None
        ),
        "best_stack_minus_current_f1": (
            best_stack_snapshot["f1"] - current_snapshot["f1"]
            if best_stack_snapshot else None
        ),
        "best_stack_minus_ensemble_sharpe": (
            best_stack_snapshot["sharpe"] - ensemble_snapshot["sharpe"]
            if best_stack_snapshot else None
        ),
        "best_stack_minus_ensemble_total_return": (
            best_stack_snapshot["total_return"] - ensemble_snapshot["total_return"]
            if best_stack_snapshot else None
        ),
        "best_stack_minus_ensemble_accuracy": (
            best_stack_snapshot["accuracy"] - ensemble_snapshot["accuracy"]
            if best_stack_snapshot else None
        ),
        "best_stack_minus_ensemble_f1": (
            best_stack_snapshot["f1"] - ensemble_snapshot["f1"]
            if best_stack_snapshot else None
        ),
        "best_stack_would_replace_current": bool(
            best_stack_result
            and model_sort_key_fn(best_stack_result) > model_sort_key_fn(current_snapshot)
        ),
    }


def aggregate_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "best_stack_replaces_current_count": int(
            frame["best_stack_would_replace_current"].sum()
        ),
        "best_stack_beats_current_sharpe_count": int(
            (frame["best_stack_minus_current_sharpe"] > 0).sum()
        ),
        "best_stack_beats_ensemble_sharpe_count": int(
            (frame["best_stack_minus_ensemble_sharpe"] > 0).sum()
        ),
        "xgboost_beats_ensemble_sharpe_count": int(
            (frame["xgboost_sharpe"] > frame["ensemble_sharpe"]).sum()
        ),
        "lightgbm_beats_ensemble_sharpe_count": int(
            (frame["lightgbm_sharpe"] > frame["ensemble_sharpe"]).sum()
        ),
        "avg_current_sharpe": float(frame["current_sharpe"].mean()),
        "avg_ensemble_sharpe": float(frame["ensemble_sharpe"].mean()),
        "avg_xgboost_sharpe": float(frame["xgboost_sharpe"].mean()),
        "avg_lightgbm_sharpe": float(frame["lightgbm_sharpe"].mean()),
        "avg_best_stack_sharpe": float(frame["best_stack_sharpe"].mean()),
        "avg_best_stack_minus_current_sharpe": float(
            frame["best_stack_minus_current_sharpe"].mean()
        ),
        "avg_best_stack_minus_current_total_return": float(
            frame["best_stack_minus_current_total_return"].mean()
        ),
        "avg_best_stack_minus_current_accuracy": float(
            frame["best_stack_minus_current_accuracy"].mean()
        ),
        "avg_best_stack_minus_current_f1": float(
            frame["best_stack_minus_current_f1"].mean()
        ),
        "avg_best_stack_minus_ensemble_sharpe": float(
            frame["best_stack_minus_ensemble_sharpe"].mean()
        ),
        "avg_best_stack_minus_ensemble_f1": float(
            frame["best_stack_minus_ensemble_f1"].mean()
        ),
    }


def available_main_tickers(repo: MarketDataRepository) -> list[str]:
    return sorted(repo.list_tickers())


def available_gift_tickers(repo: MarketDataRepository) -> list[str]:
    return [
        ticker
        for ticker in sorted(repo.list_tickers())
        if not ticker.startswith("^")
    ]


def benchmark_main_pipeline(tickers: list[str]) -> pd.DataFrame:
    repo = MarketDataRepository()
    engine = BacktestEngine(transaction_cost=0.001)
    rows = []

    for ticker in tickers:
        meta_path = MAIN_ARTIFACTS / f"{ticker.replace('.','_')}_meta.json"

        if not meta_path.exists():
            continue

        meta = load_meta(meta_path)
        frame = build_main_features(repo.load(ticker))

        if len(frame) < 250:
            continue

        logistic_builder = build_main_logistic_builder(
            meta["selected_configs"]["logistic"]
        )
        rf_builder = build_main_rf_builder(
            meta["selected_configs"]["rf"]
        )

        logistic_probabilities = main_walk_forward_probabilities(
            frame,
            logistic_builder,
        )
        rf_probabilities = main_walk_forward_probabilities(
            frame,
            rf_builder,
        )

        stack_frame = build_signal_stack_frame(
            frame,
            meta,
            logistic_probabilities,
            rf_probabilities,
        )

        best_xgboost = evaluate_meta_candidates(
            stack_frame,
            STACK_FEATURE_COLUMNS,
            build_xgboost_candidates(),
            choose_main_n_splits,
            select_main_threshold,
            main_oof_classification_metrics,
            main_model_sort_key,
            engine,
        )
        best_lightgbm = evaluate_meta_candidates(
            stack_frame,
            STACK_FEATURE_COLUMNS,
            build_lightgbm_candidates(),
            choose_main_n_splits,
            select_main_threshold,
            main_oof_classification_metrics,
            main_model_sort_key,
            engine,
        )

        rows.append(comparison_row(
            pipeline="main",
            ticker=ticker,
            frame=stack_frame,
            current_model_name=meta["recommended_model"],
            current_snapshot=meta_model_snapshot(
                meta,
                meta["recommended_model"],
            ),
            ensemble_snapshot=meta_model_snapshot(meta, "ensemble"),
            best_xgboost=best_xgboost,
            best_lightgbm=best_lightgbm,
            model_sort_key_fn=main_model_sort_key,
        ))

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def benchmark_gift_pipeline(tickers: list[str], gift_path: str) -> pd.DataFrame:
    engine = IntradayBacktestEngine(transaction_cost=0.001)
    rows = []

    for ticker in tickers:
        meta_path = GIFT_ARTIFACTS / f"{ticker.replace('.','_')}_gift_meta.json"

        if not meta_path.exists():
            continue

        meta = load_meta(meta_path)
        frame = build_training_frame(
            ticker=ticker,
            gift_path=gift_path,
        )

        if len(frame) < 120:
            continue

        logistic_builder = build_gift_logistic_builder(
            meta["selected_configs"]["logistic"]
        )
        rf_builder = build_gift_rf_builder(
            meta["selected_configs"]["rf"]
        )

        logistic_probabilities = gift_walk_forward_probabilities(
            frame,
            logistic_builder,
        )
        rf_probabilities = gift_walk_forward_probabilities(
            frame,
            rf_builder,
        )

        stack_frame = build_signal_stack_frame(
            frame,
            meta,
            logistic_probabilities,
            rf_probabilities,
        )

        best_xgboost = evaluate_meta_candidates(
            stack_frame,
            STACK_FEATURE_COLUMNS,
            build_xgboost_candidates(),
            choose_gift_n_splits,
            select_gift_threshold,
            gift_oof_classification_metrics,
            gift_model_sort_key,
            engine,
        )
        best_lightgbm = evaluate_meta_candidates(
            stack_frame,
            STACK_FEATURE_COLUMNS,
            build_lightgbm_candidates(),
            choose_gift_n_splits,
            select_gift_threshold,
            gift_oof_classification_metrics,
            gift_model_sort_key,
            engine,
        )

        rows.append(comparison_row(
            pipeline="gift",
            ticker=ticker,
            frame=stack_frame,
            current_model_name=meta["recommended_model"],
            current_snapshot=meta_model_snapshot(
                meta,
                meta["recommended_model"],
            ),
            ensemble_snapshot=meta_model_snapshot(meta, "ensemble"),
            best_xgboost=best_xgboost,
            best_lightgbm=best_lightgbm,
            model_sort_key_fn=gift_model_sort_key,
        ))

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def write_outputs(main_frame: pd.DataFrame, gift_frame: pd.DataFrame):
    summary = {}

    if not main_frame.empty:
        main_path = OUTPUT_DIR / "main_signal_stacking.csv"
        main_frame.to_csv(main_path, index=False)
        summary["main"] = aggregate_summary(main_frame)

    if not gift_frame.empty:
        gift_path = OUTPUT_DIR / "gift_signal_stacking.csv"
        gift_frame.to_csv(gift_path, index=False)
        summary["gift"] = aggregate_summary(gift_frame)

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark XGBoost and LightGBM stacks on existing model outputs.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["main", "gift", "both"],
        default="both",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Ticker to benchmark. Pass multiple times to limit the run.",
    )
    parser.add_argument(
        "--gift-path",
        default=DEFAULT_GIFT_DATA_PATH,
        help="Path to normalized GIFT Nifty OHLC CSV.",
    )

    args = parser.parse_args()

    repo = MarketDataRepository()
    selected_tickers = set(args.tickers or [])

    main_tickers = []
    gift_tickers = []

    if args.pipeline in ("main", "both"):
        main_tickers = available_main_tickers(repo)
        if selected_tickers:
            main_tickers = [
                ticker for ticker in main_tickers
                if ticker in selected_tickers
            ]

    if args.pipeline in ("gift", "both"):
        gift_tickers = available_gift_tickers(repo)
        if selected_tickers:
            gift_tickers = [
                ticker for ticker in gift_tickers
                if ticker in selected_tickers
            ]

    main_frame = benchmark_main_pipeline(main_tickers) if main_tickers else pd.DataFrame()
    gift_frame = benchmark_gift_pipeline(
        gift_tickers,
        args.gift_path,
    ) if gift_tickers else pd.DataFrame()

    summary = write_outputs(main_frame, gift_frame)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
