import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analytics.performance import (
    max_drawdown,
    sharpe_ratio,
    total_return,
    win_rate,
)
from src.gift_nifty.backtesting import IntradayBacktestEngine
from src.gift_nifty.constants import DEFAULT_GIFT_DATA_PATH
from src.gift_nifty.dataset import GIFT_MODEL_FEATURE_COLUMNS
from src.gift_nifty.train_walkforward import (
    THRESHOLD_GRID,
    BLEND_WEIGHTS,
    build_logistic_candidates,
    build_rf_candidates,
    build_training_frame,
    model_sort_key,
)
from src.models.probabilities import predict_up_probability
from src.validation.metrics import classification_metrics_from_probabilities


MAIN_ARTIFACTS = Path("artifacts/models")
GIFT_ARTIFACTS = Path("artifacts/gift_models")
REPORT_DIR = GIFT_ARTIFACTS / "comparison"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


BASELINE_FEATURE_COLUMNS = [
    column
    for column in GIFT_MODEL_FEATURE_COLUMNS
    if column.startswith("stock_prev_")
]


def load_meta(path: Path) -> dict:
    return json.loads(path.read_text())


def walk_forward_probabilities(df: pd.DataFrame, feature_columns, model_builder):
    from sklearn.model_selection import TimeSeriesSplit

    probabilities = pd.Series(index=df.index, dtype=float)

    n_splits = max(3, min(5, len(df) // 60))
    splitter = TimeSeriesSplit(n_splits=n_splits)

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


def evaluate_candidates(df, feature_columns, engine, candidates):
    best = None

    for candidate in candidates:
        probabilities = walk_forward_probabilities(
            df,
            feature_columns,
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


def summarize_best_results(best_logistic, best_rf, best_ensemble):
    metrics_by_model = {
        "logistic": best_logistic,
        "rf": best_rf,
        "ensemble": best_ensemble,
    }

    recommended_model = max(
        metrics_by_model,
        key=lambda model_name: model_sort_key(metrics_by_model[model_name]),
    )

    best = metrics_by_model[recommended_model]
    cls = best["classification_metrics"]

    return {
        "recommended_model": recommended_model,
        "sharpe": float(best["sharpe"]),
        "total_return": float(best["total_return"]),
        "max_drawdown": float(best["max_drawdown"]),
        "win_rate": float(best["win_rate"]),
        "threshold": float(best["threshold"]),
        "accuracy": float(cls["accuracy"]),
        "precision": float(cls["precision"]),
        "recall": float(cls["recall"]),
        "f1": float(cls["f1"]),
        "positive_rate": float(cls["positive_rate"]),
        "support": int(cls["support"]),
    }


def evaluate_intraday_baseline(ticker: str, gift_path: str) -> dict:
    df = build_training_frame(
        ticker=ticker,
        gift_path=gift_path,
    )

    engine = IntradayBacktestEngine(transaction_cost=0.001)

    best_logistic = evaluate_candidates(
        df,
        BASELINE_FEATURE_COLUMNS,
        engine,
        build_logistic_candidates(),
    )
    best_rf = evaluate_candidates(
        df,
        BASELINE_FEATURE_COLUMNS,
        engine,
        build_rf_candidates(),
    )
    best_ensemble = evaluate_blend(
        df,
        engine,
        best_logistic,
        best_rf,
    )

    return {
        "rows": int(len(df)),
        "start_date": str(pd.Timestamp(df["Date"].min()).date()),
        "end_date": str(pd.Timestamp(df["Date"].max()).date()),
        **summarize_best_results(
            best_logistic,
            best_rf,
            best_ensemble,
        ),
    }


def recommended_native_metrics(meta: dict) -> dict:
    model_name = meta["recommended_model"]
    oof = meta["oof_metrics"][model_name]
    cls = meta["oof_classification_metrics"][model_name]

    return {
        "recommended_model": model_name,
        "sharpe": float(oof["sharpe"]),
        "total_return": float(oof["total_return"]),
        "max_drawdown": float(oof["max_drawdown"]),
        "win_rate": float(oof["win_rate"]),
        "threshold": float(oof["threshold"]),
        "accuracy": float(cls["accuracy"]),
        "precision": float(cls["precision"]),
        "recall": float(cls["recall"]),
        "f1": float(cls["f1"]),
        "positive_rate": float(cls["positive_rate"]),
        "support": int(cls["support"]),
    }


def build_comparison_frame(gift_path: str) -> pd.DataFrame:
    rows = []

    for gift_meta_path in sorted(GIFT_ARTIFACTS.glob("*_gift_meta.json")):
        ticker = gift_meta_path.name.replace("_gift_meta.json", "").replace("_", ".")
        main_meta_path = MAIN_ARTIFACTS / f"{ticker.replace('.','_')}_meta.json"

        if not main_meta_path.exists():
            continue

        gift_meta = load_meta(gift_meta_path)
        main_meta = load_meta(main_meta_path)

        intraday_baseline = evaluate_intraday_baseline(
            ticker=ticker,
            gift_path=gift_path,
        )
        gift_native = recommended_native_metrics(gift_meta)
        main_native = recommended_native_metrics(main_meta)

        rows.append({
            "ticker": ticker,
            "comparison_rows": intraday_baseline["rows"],
            "comparison_start_date": intraday_baseline["start_date"],
            "comparison_end_date": intraday_baseline["end_date"],
            "main_native_model": main_native["recommended_model"],
            "main_native_sharpe": main_native["sharpe"],
            "main_native_total_return": main_native["total_return"],
            "main_native_max_drawdown": main_native["max_drawdown"],
            "main_native_win_rate": main_native["win_rate"],
            "main_native_accuracy": main_native["accuracy"],
            "main_native_f1": main_native["f1"],
            "baseline_model": intraday_baseline["recommended_model"],
            "baseline_sharpe": intraday_baseline["sharpe"],
            "baseline_total_return": intraday_baseline["total_return"],
            "baseline_max_drawdown": intraday_baseline["max_drawdown"],
            "baseline_win_rate": intraday_baseline["win_rate"],
            "baseline_accuracy": intraday_baseline["accuracy"],
            "baseline_f1": intraday_baseline["f1"],
            "gift_model": gift_native["recommended_model"],
            "gift_sharpe": gift_native["sharpe"],
            "gift_total_return": gift_native["total_return"],
            "gift_max_drawdown": gift_native["max_drawdown"],
            "gift_win_rate": gift_native["win_rate"],
            "gift_accuracy": gift_native["accuracy"],
            "gift_f1": gift_native["f1"],
            "gift_minus_baseline_sharpe": (
                gift_native["sharpe"] - intraday_baseline["sharpe"]
            ),
            "gift_minus_baseline_total_return": (
                gift_native["total_return"] - intraday_baseline["total_return"]
            ),
            "gift_minus_baseline_accuracy": (
                gift_native["accuracy"] - intraday_baseline["accuracy"]
            ),
            "gift_minus_baseline_f1": (
                gift_native["f1"] - intraday_baseline["f1"]
            ),
            "gift_minus_main_native_sharpe": (
                gift_native["sharpe"] - main_native["sharpe"]
            ),
            "gift_minus_main_native_total_return": (
                gift_native["total_return"] - main_native["total_return"]
            ),
            "gift_minus_main_native_accuracy": (
                gift_native["accuracy"] - main_native["accuracy"]
            ),
            "gift_minus_main_native_f1": (
                gift_native["f1"] - main_native["f1"]
            ),
        })

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)


def aggregate_summary(frame: pd.DataFrame) -> dict:
    return {
        "tickers": int(len(frame)),
        "gift_beats_baseline_sharpe_count": int(
            (frame["gift_minus_baseline_sharpe"] > 0).sum()
        ),
        "gift_beats_baseline_f1_count": int(
            (frame["gift_minus_baseline_f1"] > 0).sum()
        ),
        "avg_baseline_sharpe": float(frame["baseline_sharpe"].mean()),
        "avg_gift_sharpe": float(frame["gift_sharpe"].mean()),
        "avg_baseline_f1": float(frame["baseline_f1"].mean()),
        "avg_gift_f1": float(frame["gift_f1"].mean()),
        "avg_gift_minus_baseline_sharpe": float(
            frame["gift_minus_baseline_sharpe"].mean()
        ),
        "avg_gift_minus_baseline_total_return": float(
            frame["gift_minus_baseline_total_return"].mean()
        ),
        "avg_gift_minus_baseline_accuracy": float(
            frame["gift_minus_baseline_accuracy"].mean()
        ),
        "avg_gift_minus_baseline_f1": float(
            frame["gift_minus_baseline_f1"].mean()
        ),
        "avg_main_native_sharpe": float(frame["main_native_sharpe"].mean()),
        "avg_gift_minus_main_native_sharpe": float(
            frame["gift_minus_main_native_sharpe"].mean()
        ),
        "avg_gift_minus_main_native_f1": float(
            frame["gift_minus_main_native_f1"].mean()
        ),
    }


def build_report_figure(frame: pd.DataFrame, summary: dict) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Aligned Sharpe: Baseline vs GIFT",
            "Aligned F1: Baseline vs GIFT",
            "Native Sharpe: Main vs GIFT",
            "Aligned Sharpe Delta: GIFT - Baseline",
        ],
    )

    figure.add_trace(
        go.Bar(
            x=frame["ticker"],
            y=frame["baseline_sharpe"],
            name="Baseline Sharpe",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=frame["ticker"],
            y=frame["gift_sharpe"],
            name="GIFT Sharpe",
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Bar(
            x=frame["ticker"],
            y=frame["baseline_f1"],
            name="Baseline F1",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=frame["ticker"],
            y=frame["gift_f1"],
            name="GIFT F1",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    figure.add_trace(
        go.Bar(
            x=frame["ticker"],
            y=frame["main_native_sharpe"],
            name="Main Native Sharpe",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=frame["ticker"],
            y=frame["gift_sharpe"],
            name="GIFT Native Sharpe",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Bar(
            x=frame["ticker"],
            y=frame["gift_minus_baseline_sharpe"],
            name="GIFT - Baseline Sharpe",
            marker_color=[
                "#2e7d32" if value >= 0 else "#c62828"
                for value in frame["gift_minus_baseline_sharpe"]
            ],
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    figure.update_layout(
        barmode="group",
        height=900,
        width=1400,
        title=(
            "GIFT Comparison Report"
            f" | avg aligned Sharpe delta {summary['avg_gift_minus_baseline_sharpe']:.4f}"
            f" | avg aligned F1 delta {summary['avg_gift_minus_baseline_f1']:.4f}"
        ),
    )

    return figure


def run_comparison(gift_path: str):
    frame = build_comparison_frame(gift_path=gift_path)

    if frame.empty:
        raise ValueError("No overlapping main and GIFT artifacts were found")

    summary = aggregate_summary(frame)

    csv_path = REPORT_DIR / "comparison_summary.csv"
    json_path = REPORT_DIR / "comparison_summary.json"
    html_path = REPORT_DIR / "comparison_report.html"

    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))

    figure = build_report_figure(frame, summary)
    figure.write_html(
        html_path,
        include_plotlyjs="cdn",
        full_html=True,
    )

    print(json.dumps(summary, indent=2))
    print("csv", csv_path)
    print("json", json_path)
    print("html", html_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare standalone GIFT models against the main pipeline and an aligned intraday baseline",
    )
    parser.add_argument(
        "--gift-path",
        default=DEFAULT_GIFT_DATA_PATH,
        help="Path to normalized GIFT Nifty CSV used by the standalone models.",
    )

    args = parser.parse_args()
    run_comparison(gift_path=args.gift_path)
