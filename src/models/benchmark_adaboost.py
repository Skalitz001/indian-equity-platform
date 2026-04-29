import argparse
import json
from pathlib import Path

import pandas as pd

from src.backtesting.engine import BacktestEngine
from src.gift_nifty.backtesting import IntradayBacktestEngine
from src.gift_nifty.constants import DEFAULT_GIFT_DATA_PATH
from src.gift_nifty.train_walkforward import (
    build_logistic_candidates as build_gift_logistic_candidates,
    build_rf_candidates as build_gift_rf_candidates,
    build_training_frame,
    evaluate_blend as evaluate_gift_blend,
    evaluate_candidates as evaluate_gift_candidates,
    model_sort_key as gift_model_sort_key,
)
from src.models.boosting_candidates import build_adaboost_candidates
from src.models.train_walkforward import (
    build_features as build_main_features,
    build_logistic_candidates as build_main_logistic_candidates,
    build_rf_candidates as build_main_rf_candidates,
    evaluate_blend as evaluate_main_blend,
    evaluate_candidates as evaluate_main_candidates,
    model_sort_key as main_model_sort_key,
)
from src.repositories.market_data_repository import MarketDataRepository


OUTPUT_DIR = Path("artifacts/boosting_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAIN_ARTIFACTS = Path("artifacts/models")
GIFT_ARTIFACTS = Path("artifacts/gift_models")


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text())


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


def choose_current_winner(best_logistic, best_rf, best_ensemble, model_sort_key):
    candidates = {
        "logistic": best_logistic,
        "rf": best_rf,
        "ensemble": best_ensemble,
    }

    winner_name = max(
        candidates,
        key=lambda model_name: model_sort_key(candidates[model_name]),
    )

    return winner_name, candidates[winner_name]


def comparison_row(
    pipeline: str,
    ticker: str,
    frame: pd.DataFrame,
    current_model_name: str,
    current_snapshot: dict,
    ensemble_snapshot: dict,
    adaboost_result: dict,
    model_sort_key,
) -> dict:
    adaboost_snapshot = result_snapshot(adaboost_result)

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
        "current_max_drawdown": current_snapshot["max_drawdown"],
        "current_win_rate": current_snapshot["win_rate"],
        "current_accuracy": current_snapshot["accuracy"],
        "current_f1": current_snapshot["f1"],
        "ensemble_label": ensemble_snapshot["label"],
        "ensemble_sharpe": ensemble_snapshot["sharpe"],
        "ensemble_total_return": ensemble_snapshot["total_return"],
        "ensemble_accuracy": ensemble_snapshot["accuracy"],
        "ensemble_f1": ensemble_snapshot["f1"],
        "adaboost_label": adaboost_snapshot["label"],
        "adaboost_sharpe": adaboost_snapshot["sharpe"],
        "adaboost_total_return": adaboost_snapshot["total_return"],
        "adaboost_max_drawdown": adaboost_snapshot["max_drawdown"],
        "adaboost_win_rate": adaboost_snapshot["win_rate"],
        "adaboost_accuracy": adaboost_snapshot["accuracy"],
        "adaboost_f1": adaboost_snapshot["f1"],
        "adaboost_threshold": adaboost_snapshot["threshold"],
        "adaboost_entries": adaboost_snapshot["entries"],
        "adaboost_active_days": adaboost_snapshot["active_days"],
        "adaboost_minus_current_sharpe": (
            adaboost_snapshot["sharpe"] - current_snapshot["sharpe"]
        ),
        "adaboost_minus_current_total_return": (
            adaboost_snapshot["total_return"] - current_snapshot["total_return"]
        ),
        "adaboost_minus_current_accuracy": (
            adaboost_snapshot["accuracy"] - current_snapshot["accuracy"]
        ),
        "adaboost_minus_current_f1": (
            adaboost_snapshot["f1"] - current_snapshot["f1"]
        ),
        "adaboost_minus_ensemble_sharpe": (
            adaboost_snapshot["sharpe"] - ensemble_snapshot["sharpe"]
        ),
        "adaboost_minus_ensemble_total_return": (
            adaboost_snapshot["total_return"] - ensemble_snapshot["total_return"]
        ),
        "adaboost_minus_ensemble_accuracy": (
            adaboost_snapshot["accuracy"] - ensemble_snapshot["accuracy"]
        ),
        "adaboost_minus_ensemble_f1": (
            adaboost_snapshot["f1"] - ensemble_snapshot["f1"]
        ),
        "adaboost_would_replace_current": bool(
            model_sort_key(adaboost_result) > model_sort_key(current_snapshot)
        ),
    }


def aggregate_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "adaboost_replaces_current_count": int(
            frame["adaboost_would_replace_current"].sum()
        ),
        "adaboost_beats_current_sharpe_count": int(
            (frame["adaboost_minus_current_sharpe"] > 0).sum()
        ),
        "adaboost_beats_ensemble_sharpe_count": int(
            (frame["adaboost_minus_ensemble_sharpe"] > 0).sum()
        ),
        "avg_current_sharpe": float(frame["current_sharpe"].mean()),
        "avg_ensemble_sharpe": float(frame["ensemble_sharpe"].mean()),
        "avg_adaboost_sharpe": float(frame["adaboost_sharpe"].mean()),
        "avg_adaboost_minus_current_sharpe": float(
            frame["adaboost_minus_current_sharpe"].mean()
        ),
        "avg_adaboost_minus_current_total_return": float(
            frame["adaboost_minus_current_total_return"].mean()
        ),
        "avg_adaboost_minus_current_accuracy": float(
            frame["adaboost_minus_current_accuracy"].mean()
        ),
        "avg_adaboost_minus_current_f1": float(
            frame["adaboost_minus_current_f1"].mean()
        ),
        "avg_adaboost_minus_ensemble_sharpe": float(
            frame["adaboost_minus_ensemble_sharpe"].mean()
        ),
        "avg_adaboost_minus_ensemble_f1": float(
            frame["adaboost_minus_ensemble_f1"].mean()
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
        frame = build_main_features(repo.load(ticker))

        if len(frame) < 250:
            continue

        best_adaboost = evaluate_main_candidates(
            frame,
            engine,
            build_adaboost_candidates(),
        )

        if best_adaboost is None:
            continue

        meta_path = MAIN_ARTIFACTS / f"{ticker.replace('.','_')}_meta.json"

        if meta_path.exists():
            meta = load_meta(meta_path)
            current_model_name = meta["recommended_model"]
            current_snapshot = meta_model_snapshot(meta, current_model_name)
            ensemble_snapshot = meta_model_snapshot(meta, "ensemble")
        else:
            best_logistic = evaluate_main_candidates(
                frame,
                engine,
                build_main_logistic_candidates(),
            )
            best_rf = evaluate_main_candidates(
                frame,
                engine,
                build_main_rf_candidates(),
            )
            best_ensemble = evaluate_main_blend(
                frame,
                engine,
                best_logistic,
                best_rf,
            )
            current_model_name, current_result = choose_current_winner(
                best_logistic,
                best_rf,
                best_ensemble,
                main_model_sort_key,
            )
            current_snapshot = result_snapshot(current_result)
            ensemble_snapshot = result_snapshot(best_ensemble)

        rows.append(comparison_row(
            pipeline="main",
            ticker=ticker,
            frame=frame,
            current_model_name=current_model_name,
            current_snapshot=current_snapshot,
            ensemble_snapshot=ensemble_snapshot,
            adaboost_result=best_adaboost,
            model_sort_key=main_model_sort_key,
        ))

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def benchmark_gift_pipeline(tickers: list[str], gift_path: str) -> pd.DataFrame:
    engine = IntradayBacktestEngine(transaction_cost=0.001)
    rows = []

    for ticker in tickers:
        frame = build_training_frame(
            ticker=ticker,
            gift_path=gift_path,
        )

        if len(frame) < 120:
            continue

        best_adaboost = evaluate_gift_candidates(
            frame,
            engine,
            build_adaboost_candidates(),
        )

        if best_adaboost is None:
            continue

        meta_path = GIFT_ARTIFACTS / f"{ticker.replace('.','_')}_gift_meta.json"

        if meta_path.exists():
            meta = load_meta(meta_path)
            current_model_name = meta["recommended_model"]
            current_snapshot = meta_model_snapshot(meta, current_model_name)
            ensemble_snapshot = meta_model_snapshot(meta, "ensemble")
        else:
            best_logistic = evaluate_gift_candidates(
                frame,
                engine,
                build_gift_logistic_candidates(),
            )
            best_rf = evaluate_gift_candidates(
                frame,
                engine,
                build_gift_rf_candidates(),
            )
            best_ensemble = evaluate_gift_blend(
                frame,
                engine,
                best_logistic,
                best_rf,
            )
            current_model_name, current_result = choose_current_winner(
                best_logistic,
                best_rf,
                best_ensemble,
                gift_model_sort_key,
            )
            current_snapshot = result_snapshot(current_result)
            ensemble_snapshot = result_snapshot(best_ensemble)

        rows.append(comparison_row(
            pipeline="gift",
            ticker=ticker,
            frame=frame,
            current_model_name=current_model_name,
            current_snapshot=current_snapshot,
            ensemble_snapshot=ensemble_snapshot,
            adaboost_result=best_adaboost,
            model_sort_key=gift_model_sort_key,
        ))

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def write_outputs(main_frame: pd.DataFrame, gift_frame: pd.DataFrame):
    summary = {}

    if not main_frame.empty:
        main_path = OUTPUT_DIR / "main_adaboost_vs_baseline.csv"
        main_frame.to_csv(main_path, index=False)
        summary["main"] = aggregate_summary(main_frame)

    if not gift_frame.empty:
        gift_path = OUTPUT_DIR / "gift_adaboost_vs_baseline.csv"
        gift_frame.to_csv(gift_path, index=False)
        summary["gift"] = aggregate_summary(gift_frame)

    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark AdaBoost against current model families.",
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
