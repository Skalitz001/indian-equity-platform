import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.analytics.performance import (
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)
from src.gift_nifty.backtesting import IntradayBacktestEngine
from src.gift_nifty.constants import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_GIFT_DATA_PATH,
)
from src.gift_nifty.dataset import (
    GIFT_MODEL_FEATURE_COLUMNS,
    GIFT_PRE_OPEN_SOURCE_ASSUMPTION,
    GIFT_STOCK_FEATURE_GROUP,
    build_gift_model_frame,
)
from src.gift_nifty.repository import GiftNiftyRepository
from src.models.probabilities import predict_up_probability
from src.repositories.market_data_repository import MarketDataRepository
from src.validation.metrics import classification_metrics_from_probabilities
from src.validation.prediction_store import (
    build_prediction_frame,
    save_prediction_frame,
)


ARTIFACTS = Path(DEFAULT_ARTIFACTS_DIR)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

FEATURES = GIFT_MODEL_FEATURE_COLUMNS
THRESHOLD_GRID = np.arange(0.50, 0.81, 0.01)
BLEND_WEIGHTS = (0.35, 0.50, 0.65)
HOLDOUT_FRACTION = 0.20
MIN_HOLDOUT_ROWS = 60
MIN_SELECTION_ROWS = 120


def choose_n_splits(df: pd.DataFrame) -> int:
    return max(3, min(5, len(df) // 60))


def split_selection_holdout(
    df: pd.DataFrame,
    min_selection_rows: int = MIN_SELECTION_ROWS,
    min_holdout_rows: int = MIN_HOLDOUT_ROWS,
    holdout_fraction: float = HOLDOUT_FRACTION,
):
    max_holdout_rows = len(df) - min_selection_rows

    if max_holdout_rows < min_holdout_rows:
        return df.copy(), df.iloc[0:0].copy()

    holdout_rows = min(
        max(min_holdout_rows, int(len(df) * holdout_fraction)),
        max_holdout_rows,
    )

    return (
        df.iloc[:-holdout_rows].copy().reset_index(drop=True),
        df.iloc[-holdout_rows:].copy().reset_index(drop=True),
    )


def walk_forward_probabilities(df: pd.DataFrame, model_builder):
    probabilities = pd.Series(index=df.index, dtype=float)

    splitter = TimeSeriesSplit(
        n_splits=choose_n_splits(df)
    )

    for train_idx, test_idx in splitter.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = model_builder()
        model.fit(
            train_df[FEATURES],
            train_df["target"],
        )

        probabilities.iloc[test_idx] = predict_up_probability(
            model,
            test_df[FEATURES],
        )

    return probabilities.dropna()


def signal_policy_metrics(df, probabilities, threshold, engine):
    signals = (pd.Series(probabilities) > float(threshold)).astype(int)

    bt = engine.run(
        df.loc[signals.index],
        signals,
    )

    returns = bt["strategy_return"].dropna()

    if len(returns) < 5:
        return None

    active_days = int(signals.sum())

    return {
        "threshold": float(threshold),
        "entry_threshold": float(threshold),
        "exit_threshold": float(threshold),
        "sharpe": float(sharpe_ratio(returns)),
        "total_return": float(total_return(bt["equity_curve"])),
        "max_drawdown": float(max_drawdown(bt["equity_curve"])),
        "win_rate": float(win_rate(returns)),
        "active_days": active_days,
        "entries": active_days,
    }


def select_threshold(df, probabilities, engine):
    min_active_days = max(10, int(len(probabilities) * 0.02))

    best = None

    for threshold in THRESHOLD_GRID:
        metrics = signal_policy_metrics(
            df,
            probabilities,
            threshold,
            engine,
        )

        if metrics is None:
            continue

        if metrics["active_days"] < min_active_days:
            continue

        if best is None or model_sort_key(metrics) > model_sort_key(best):
            best = metrics

    if best is not None:
        return best

    for threshold in THRESHOLD_GRID:
        metrics = signal_policy_metrics(
            df,
            probabilities,
            threshold,
            engine,
        )

        if metrics is None:
            continue

        if best is None or model_sort_key(metrics) > model_sort_key(best):
            best = metrics

    return best


def oof_classification_metrics(df, probabilities):
    return classification_metrics_from_probabilities(
        df.loc[probabilities.index, "target"],
        probabilities,
        optimize_threshold=True,
    )


def model_sort_key(result):
    return (
        result["sharpe"],
        result["total_return"],
        result["max_drawdown"],
        result["win_rate"],
        -result.get("entries", 0),
        result["active_days"],
    )


def build_logistic_candidates():
    candidates = []

    for c in (0.1, 0.3, 1.0, 3.0):
        for class_weight in ("balanced", None):
            config = {
                "C": c,
                "class_weight": class_weight,
            }

            label = (
                f"logreg_c{c}_"
                f"{class_weight or 'none'}"
            )

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

            candidates.append({
                "label": label,
                "config": config,
                "builder": model_builder,
            })

    return candidates


def build_rf_candidates():
    candidates = []

    for n_estimators, max_depth, min_leaf, max_features in (
        (250, 4, 10, "sqrt"),
        (400, 5, 10, "sqrt"),
        (400, 6, 20, "sqrt"),
        (600, 6, 10, "sqrt"),
        (400, 6, 10, 0.5),
        (600, 8, 20, 0.5),
    ):
        config = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_leaf,
            "max_features": max_features,
        }

        label = (
            f"rf_ne{n_estimators}_"
            f"md{max_depth}_"
            f"ml{min_leaf}_"
            f"mf{max_features}"
        )

        def model_builder(config=config):
            return RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                min_samples_leaf=config["min_samples_leaf"],
                max_features=config["max_features"],
                random_state=42,
            )

        candidates.append({
            "label": label,
            "config": config,
            "builder": model_builder,
        })

    return candidates


def evaluate_candidates(df, engine, candidates):
    best = None

    for candidate in candidates:
        probabilities = walk_forward_probabilities(
            df,
            candidate["builder"],
        )

        if probabilities.empty:
            continue

        metrics = select_threshold(
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
            "builder": candidate["builder"],
            "classification_metrics": oof_classification_metrics(
                df,
                probabilities,
            ),
            "probabilities": probabilities,
        }

        if best is None or model_sort_key(result) > model_sort_key(best):
            best = result

    return best


def evaluate_blend(df, engine, logistic_result, rf_result):
    best = None

    common_index = logistic_result["probabilities"].index.intersection(
        rf_result["probabilities"].index
    )

    log_probs = logistic_result["probabilities"].loc[common_index]
    rf_probs = rf_result["probabilities"].loc[common_index]

    for log_weight in BLEND_WEIGHTS:
        probabilities = (
            log_weight * log_probs
            + (1 - log_weight) * rf_probs
        )

        metrics = select_threshold(
            df,
            probabilities,
            engine,
        )

        if metrics is None:
            continue

        result = {
            **metrics,
            "label": f"ensemble_{log_weight:.2f}",
            "weights": {
                "logistic": float(log_weight),
                "rf": float(1 - log_weight),
            },
            "classification_metrics": oof_classification_metrics(
                df,
                probabilities,
            ),
            "probabilities": probabilities,
        }

        if best is None or model_sort_key(result) > model_sort_key(best):
            best = result

    return best


def fit_final_model(df, candidate):
    model = candidate["builder"]()
    model.fit(
        df[FEATURES],
        df["target"],
    )
    return model


def evaluate_model_holdout(
    holdout_df,
    model,
    threshold,
    engine,
):
    if holdout_df.empty:
        return None, None, None

    probabilities = pd.Series(
        predict_up_probability(model, holdout_df[FEATURES]),
        index=holdout_df.index,
        dtype=float,
    )
    signals = (probabilities > float(threshold)).astype(int)
    bt = engine.run(holdout_df, signals)
    returns = bt["strategy_return"].dropna()

    if len(returns) < 5:
        return None, None, None

    equity_curve = (1 + returns).cumprod()
    metrics = {
        "sharpe": float(sharpe_ratio(returns)),
        "total_return": float(total_return(equity_curve)),
        "max_drawdown": float(max_drawdown(equity_curve)),
        "win_rate": float(win_rate(returns)),
        "active_days": int(signals.sum()),
        "entries": int(signals.sum()),
        "observations": int(len(returns)),
    }

    return metrics, probabilities, signals


def build_oof_prediction_frame(
    ticker,
    model_name,
    result,
    selection_df,
):
    probabilities = result["probabilities"]
    threshold = float(result["threshold"])
    signals = (pd.Series(probabilities) > threshold).astype(int)

    return build_prediction_frame(
        selection_df,
        probabilities,
        signals,
        ticker=ticker,
        pipeline="gift",
        model_name=model_name,
        model_label=result["label"],
        feature_group=GIFT_STOCK_FEATURE_GROUP,
        split="oof_selection",
        entry_threshold=threshold,
        exit_threshold=threshold,
        return_column="intraday_return",
    )


def build_holdout_prediction_frame(
    ticker,
    model_name,
    result,
    holdout_df,
    probabilities,
    signals,
):
    threshold = float(result["threshold"])

    return build_prediction_frame(
        holdout_df,
        probabilities,
        signals,
        ticker=ticker,
        pipeline="gift",
        model_name=model_name,
        model_label=result["label"],
        feature_group=GIFT_STOCK_FEATURE_GROUP,
        split="holdout",
        entry_threshold=threshold,
        exit_threshold=threshold,
        return_column="intraday_return",
    )


def meta_metrics(result):
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
    }


def build_training_frame(ticker: str, gift_path: str) -> pd.DataFrame:
    repo = MarketDataRepository()
    gift_repo = GiftNiftyRepository(gift_path)

    stock_df = repo.load(ticker)
    gift_df = gift_repo.load()

    return build_gift_model_frame(
        stock_df=stock_df,
        gift_df=gift_df,
    )


def train_ticker(
    ticker: str,
    gift_path: str = DEFAULT_GIFT_DATA_PATH,
):
    print("Training GIFT-aware model for", ticker)

    df = build_training_frame(
        ticker=ticker,
        gift_path=gift_path,
    )

    if len(df) < 120:
        print("Not enough post-2023 GIFT-aligned data")
        return

    selection_df, holdout_df = split_selection_holdout(df)

    print(
        "Selection rows",
        len(selection_df),
        "| holdout rows",
        len(holdout_df),
    )

    engine = IntradayBacktestEngine(transaction_cost=0.001)

    best_logistic = evaluate_candidates(
        selection_df,
        engine,
        build_logistic_candidates(),
    )
    best_rf = evaluate_candidates(
        selection_df,
        engine,
        build_rf_candidates(),
    )

    if best_logistic is None or best_rf is None:
        print("Unable to fit GIFT-aware candidate models")
        return

    best_ensemble = evaluate_blend(
        selection_df,
        engine,
        best_logistic,
        best_rf,
    )

    if best_ensemble is None:
        print("Unable to fit GIFT-aware ensemble")
        return

    model_results = {
        "logistic": best_logistic,
        "rf": best_rf,
        "ensemble": best_ensemble,
    }

    holdout_metrics = {}
    prediction_store_paths = {}

    if not holdout_df.empty:
        holdout_logistic_model = fit_final_model(selection_df, best_logistic)
        holdout_rf_model = fit_final_model(selection_df, best_rf)
        holdout_models = {
            "logistic": holdout_logistic_model,
            "rf": holdout_rf_model,
            "ensemble": {
                "kind": "ensemble",
                "log_model": holdout_logistic_model,
                "rf_model": holdout_rf_model,
                "weights": best_ensemble["weights"],
            },
        }

        for model_name, result in model_results.items():
            metrics, holdout_probabilities, holdout_signals = evaluate_model_holdout(
                holdout_df,
                holdout_models[model_name],
                float(result["threshold"]),
                engine,
            )

            if metrics is not None:
                holdout_metrics[model_name] = metrics
                prediction_frame = pd.concat(
                    [
                        build_oof_prediction_frame(
                            ticker,
                            model_name,
                            result,
                            selection_df,
                        ),
                        build_holdout_prediction_frame(
                            ticker,
                            model_name,
                            result,
                            holdout_df,
                            holdout_probabilities,
                            holdout_signals,
                        ),
                    ],
                    ignore_index=True,
                )
            else:
                prediction_frame = build_oof_prediction_frame(
                    ticker,
                    model_name,
                    result,
                    selection_df,
                )

            prediction_store_paths[model_name] = str(
                save_prediction_frame(
                    prediction_frame,
                    artifacts_dir=ARTIFACTS,
                    ticker=ticker,
                    pipeline="gift",
                    model_name=model_name,
                )
            )
    else:
        for model_name, result in model_results.items():
            prediction_store_paths[model_name] = str(
                save_prediction_frame(
                    build_oof_prediction_frame(
                        ticker,
                        model_name,
                        result,
                        selection_df,
                    ),
                    artifacts_dir=ARTIFACTS,
                    ticker=ticker,
                    pipeline="gift",
                    model_name=model_name,
                )
            )

    logistic_model = fit_final_model(df, best_logistic)
    rf_model = fit_final_model(df, best_rf)

    safe_ticker = ticker.replace(".", "_")
    log_name = f"{safe_ticker}_gift_logreg.joblib"
    rf_name = f"{safe_ticker}_gift_rf.joblib"
    meta_name = f"{safe_ticker}_gift_meta.json"

    dump(logistic_model, ARTIFACTS / log_name)
    dump(rf_model, ARTIFACTS / rf_name)

    thresholds = {
        "logistic": float(best_logistic["threshold"]),
        "rf": float(best_rf["threshold"]),
        "ensemble": float(best_ensemble["threshold"]),
    }

    signal_policies = {
        "logistic": {
            "entry_threshold": float(best_logistic["entry_threshold"]),
            "exit_threshold": float(best_logistic["exit_threshold"]),
        },
        "rf": {
            "entry_threshold": float(best_rf["entry_threshold"]),
            "exit_threshold": float(best_rf["exit_threshold"]),
        },
        "ensemble": {
            "entry_threshold": float(best_ensemble["entry_threshold"]),
            "exit_threshold": float(best_ensemble["exit_threshold"]),
        },
    }

    oof_metrics = {
        model_name: meta_metrics(result)
        for model_name, result in model_results.items()
    }

    oof_classification_metrics_by_model = {
        model_name: result["classification_metrics"]
        for model_name, result in model_results.items()
    }

    recommended_model = max(
        oof_metrics,
        key=lambda model_name: model_sort_key(
            oof_metrics[model_name]
        ),
    )

    meta = {
        "ticker": ticker,
        "module": "gift_nifty",
        "training_window": {
            "start_date": str(pd.Timestamp(df["Date"].min()).date()),
            "end_date": str(pd.Timestamp(df["Date"].max()).date()),
            "rows": int(len(df)),
        },
        "target_definition": "same_day_intraday_direction",
        "execution_assumption": "enter_on_open_exit_on_close",
        "gift_input": {
            "source_path": gift_path,
            "source_date_column": "Date",
            "minimum_date": "2023-07-03",
            "availability_assumption": GIFT_PRE_OPEN_SOURCE_ASSUMPTION,
        },
        "stock_feature_group": GIFT_STOCK_FEATURE_GROUP,
        "features": FEATURES,
        "log_model": log_name,
        "rf_model": rf_name,
        "opt_threshold": float(best_logistic["threshold"]),
        "thresholds": thresholds,
        "signal_policies": signal_policies,
        "oof_metrics": oof_metrics,
        "holdout_metrics": holdout_metrics,
        "oof_classification_metrics": oof_classification_metrics_by_model,
        "recommended_model": recommended_model,
        "split_summary": {
            "selection_rows": int(len(selection_df)),
            "holdout_rows": int(len(holdout_df)),
            "holdout_fraction": float(HOLDOUT_FRACTION),
            "holdout_start_date": (
                str(pd.Timestamp(holdout_df["Date"].min()).date())
                if not holdout_df.empty else None
            ),
            "holdout_end_date": (
                str(pd.Timestamp(holdout_df["Date"].max()).date())
                if not holdout_df.empty else None
            ),
        },
        "prediction_store": prediction_store_paths,
        "selected_configs": {
            "logistic": best_logistic["config"],
            "rf": best_rf["config"],
        },
        "blend_weights": best_ensemble["weights"],
    }

    with open(ARTIFACTS / meta_name, "w") as handle:
        json.dump(meta, handle, indent=2)

    print(
        "Selected",
        recommended_model,
        "| logistic sharpe",
        round(best_logistic["sharpe"], 3),
        "| rf sharpe",
        round(best_rf["sharpe"], 3),
        "| ensemble sharpe",
        round(best_ensemble["sharpe"], 3),
    )


def run_all(gift_path: str):
    repo = MarketDataRepository()

    for ticker in repo.list_tickers():
        if ticker.startswith("^"):
            continue

        train_ticker(
            ticker=ticker,
            gift_path=gift_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train standalone GIFT-aware walk-forward models",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Ticker to train. Pass multiple times for multiple tickers.",
    )
    parser.add_argument(
        "--gift-path",
        default=DEFAULT_GIFT_DATA_PATH,
        help="Path to normalized post-2023 GIFT Nifty OHLC CSV.",
    )

    args = parser.parse_args()

    if args.tickers:
        for ticker in args.tickers:
            train_ticker(
                ticker=ticker,
                gift_path=args.gift_path,
            )
    else:
        run_all(gift_path=args.gift_path)
