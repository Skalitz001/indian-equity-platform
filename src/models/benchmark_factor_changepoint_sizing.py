import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.factor_changepoint_features import FACTOR_PREFIX
from src.analytics.features import FEATURE_COLUMNS
from src.analytics.performance import max_drawdown, sharpe_ratio, total_return, win_rate
from src.backtesting.engine import BacktestEngine
from src.models.benchmark_factor_changepoint_features import (
    FACTOR_FEATURE_GROUPS,
    build_rf_builder,
    build_training_frame,
    prepare_universe_feature_frames,
    walk_forward_probabilities,
)
from src.models.train_walkforward import model_sort_key, select_threshold
from src.repositories.market_data_repository import MarketDataRepository
from src.validation.metrics import classification_metrics_from_probabilities


ARTIFACTS_DIR = Path("artifacts/models")
OUTPUT_DIR = Path("artifacts/factor_changepoint_sizing_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSTANTIAL_SHARPE_DELTA = 0.003
SUBSTANTIAL_IMPROVED_RATE = 0.60

FEATURE_GROUP_NAME = "factor_residual_context_cp"
FEATURE_COLUMNS_FOR_MODEL = [
    *FEATURE_COLUMNS,
    *FACTOR_FEATURE_GROUPS[FEATURE_GROUP_NAME],
]

ENTRY_THRESHOLDS = (0.50, 0.53, 0.56, 0.59, 0.62)
EDGE_POWERS = (0.5, 1.0, 1.5, 2.0)
REGIME_POWERS = (0.0, 0.5, 1.0)
BREAK_PENALTIES = (0.0, 0.5, 1.0)
SYSTEMIC_PENALTIES = (0.0, 0.5)
VOL_PENALTIES = (0.0, 0.5, 1.0)
MAX_POSITIONS = (0.5, 0.75, 1.0)


def load_meta(ticker: str) -> dict:
    meta_path = ARTIFACTS_DIR / f"{ticker.replace('.','_')}_meta.json"
    return json.loads(meta_path.read_text())


def fractional_position_signals(
    df: pd.DataFrame,
    probabilities: pd.Series,
    policy: dict,
) -> pd.Series:
    aligned = df.loc[probabilities.index]
    probabilities = pd.Series(probabilities).astype(float)

    threshold = float(policy["entry_threshold"])
    edge_power = float(policy["edge_power"])
    regime_power = float(policy["regime_power"])
    break_penalty = float(policy["break_penalty"])
    systemic_penalty = float(policy["systemic_penalty"])
    vol_penalty = float(policy["vol_penalty"])
    max_position = float(policy["max_position"])

    edge = ((probabilities - threshold) / max(1.0 - threshold, 1e-12)).clip(
        lower=0.0,
        upper=1.0,
    )
    conviction = edge.pow(edge_power)

    regime_confidence = aligned[f"{FACTOR_PREFIX}_regime_confidence"].astype(float)
    regime_confidence = regime_confidence.fillna(0.0).clip(0.0, 1.0)
    if regime_power > 0:
        conviction = conviction * regime_confidence.pow(regime_power)

    unexplained_break = aligned[f"{FACTOR_PREFIX}_unexplained_break"].astype(float)
    unexplained_break = unexplained_break.fillna(0.0).clip(0.0, 1.0)
    systemic_break = aligned[
        f"{FACTOR_PREFIX}_breadth_weighted_systemic_pressure"
    ].astype(float)
    systemic_break = systemic_break.fillna(0.0).clip(0.0, 1.0)
    residual_vol_ratio = aligned[
        f"{FACTOR_PREFIX}_resid_vol_ratio_20_60"
    ].astype(float)
    residual_vol_ratio = residual_vol_ratio.fillna(1.0).clip(lower=0.25, upper=4.0)

    break_scale = (1.0 - (break_penalty * unexplained_break)).clip(0.0, 1.0)
    systemic_scale = (1.0 - (systemic_penalty * systemic_break)).clip(0.0, 1.0)
    vol_scale = 1.0 / (
        1.0 + (vol_penalty * (residual_vol_ratio - 1.0).clip(lower=0.0))
    )

    position = (
        conviction
        * break_scale
        * systemic_scale
        * vol_scale
        * max_position
    )

    return position.clip(lower=0.0, upper=1.0).fillna(0.0)


def signal_metrics(
    df: pd.DataFrame,
    signals: pd.Series,
    engine: BacktestEngine,
) -> dict | None:
    bt = engine.run(
        df.loc[signals.index],
        signals,
    )
    returns = bt["strategy_return"].dropna()

    if len(returns) < 5:
        return None

    active_days = int((signals > 0.01).sum())
    entries = int(
        (signals.gt(0.01).astype(int).diff().clip(lower=0).fillna(
            signals.iloc[0] > 0.01
        )).sum()
    )

    return {
        "sharpe": float(sharpe_ratio(returns)),
        "total_return": float(total_return(bt["equity_curve"])),
        "max_drawdown": float(max_drawdown(bt["equity_curve"])),
        "win_rate": float(win_rate(returns)),
        "active_days": active_days,
        "entries": entries,
        "avg_position": float(signals.mean()),
        "max_position_seen": float(signals.max()),
    }


def select_fractional_policy(
    df: pd.DataFrame,
    probabilities: pd.Series,
    engine: BacktestEngine,
) -> dict | None:
    min_active_days = max(20, int(len(probabilities) * 0.02))
    min_entries = max(5, int(len(probabilities) * 0.005))
    best = None

    for entry_threshold in ENTRY_THRESHOLDS:
        for edge_power in EDGE_POWERS:
            for regime_power in REGIME_POWERS:
                for break_penalty in BREAK_PENALTIES:
                    for systemic_penalty in SYSTEMIC_PENALTIES:
                        for vol_penalty in VOL_PENALTIES:
                            for max_position in MAX_POSITIONS:
                                policy = {
                                    "entry_threshold": entry_threshold,
                                    "edge_power": edge_power,
                                    "regime_power": regime_power,
                                    "break_penalty": break_penalty,
                                    "systemic_penalty": systemic_penalty,
                                    "vol_penalty": vol_penalty,
                                    "max_position": max_position,
                                }
                                signals = fractional_position_signals(
                                    df,
                                    probabilities,
                                    policy,
                                )
                                metrics = signal_metrics(df, signals, engine)

                                if metrics is None:
                                    continue

                                if metrics["active_days"] < min_active_days:
                                    continue

                                if metrics["entries"] < min_entries:
                                    continue

                                result = {
                                    **metrics,
                                    **policy,
                                    "threshold": float(entry_threshold),
                                    "entry_threshold": float(entry_threshold),
                                    "exit_threshold": float(entry_threshold),
                                }

                                if best is None or model_sort_key(result) > model_sort_key(best):
                                    best = result

    return best


def benchmark_ticker(
    ticker: str,
    feature_frames: dict[str, pd.DataFrame],
) -> dict | None:
    meta = load_meta(ticker)
    frame = build_training_frame(ticker, feature_frames)

    if len(frame) < 250:
        return None

    engine = BacktestEngine(transaction_cost=0.001)
    rf_config = meta["selected_configs"]["rf"]

    baseline_probabilities = walk_forward_probabilities(
        frame,
        FEATURE_COLUMNS,
        build_rf_builder(rf_config),
    )
    candidate_probabilities = walk_forward_probabilities(
        frame,
        FEATURE_COLUMNS_FOR_MODEL,
        build_rf_builder(rf_config),
    )

    baseline = select_threshold(
        frame,
        baseline_probabilities,
        engine,
    )
    candidate = select_fractional_policy(
        frame,
        candidate_probabilities,
        engine,
    )

    if baseline is None or candidate is None:
        return None

    baseline_classification = classification_metrics_from_probabilities(
        frame.loc[baseline_probabilities.index, "target"],
        baseline_probabilities,
        optimize_threshold=True,
    )
    candidate_classification = classification_metrics_from_probabilities(
        frame.loc[candidate_probabilities.index, "target"],
        candidate_probabilities,
        optimize_threshold=True,
    )

    return {
        "ticker": ticker,
        "rows": int(len(frame)),
        "start_date": str(pd.Timestamp(frame["Date"].min()).date()),
        "end_date": str(pd.Timestamp(frame["Date"].max()).date()),
        "baseline_f1": float(baseline_classification["f1"]),
        "candidate_f1": float(candidate_classification["f1"]),
        "delta_f1": float(
            candidate_classification["f1"] - baseline_classification["f1"]
        ),
        "baseline_accuracy": float(baseline_classification["accuracy"]),
        "candidate_accuracy": float(candidate_classification["accuracy"]),
        "delta_accuracy": float(
            candidate_classification["accuracy"]
            - baseline_classification["accuracy"]
        ),
        "baseline_sharpe": float(baseline["sharpe"]),
        "candidate_sharpe": float(candidate["sharpe"]),
        "delta_sharpe": float(candidate["sharpe"] - baseline["sharpe"]),
        "baseline_total_return": float(baseline["total_return"]),
        "candidate_total_return": float(candidate["total_return"]),
        "delta_total_return": float(
            candidate["total_return"] - baseline["total_return"]
        ),
        "baseline_max_drawdown": float(baseline["max_drawdown"]),
        "candidate_max_drawdown": float(candidate["max_drawdown"]),
        "delta_max_drawdown": float(
            candidate["max_drawdown"] - baseline["max_drawdown"]
        ),
        "baseline_win_rate": float(baseline["win_rate"]),
        "candidate_win_rate": float(candidate["win_rate"]),
        "delta_win_rate": float(candidate["win_rate"] - baseline["win_rate"]),
        "baseline_active_days": int(baseline["active_days"]),
        "candidate_active_days": int(candidate["active_days"]),
        "delta_active_days": int(candidate["active_days"] - baseline["active_days"]),
        "baseline_entries": int(baseline["entries"]),
        "candidate_entries": int(candidate["entries"]),
        "delta_entries": int(candidate["entries"] - baseline["entries"]),
        "candidate_avg_position": float(candidate["avg_position"]),
        "candidate_max_position_seen": float(candidate["max_position_seen"]),
        "candidate_entry_threshold": float(candidate["entry_threshold"]),
        "candidate_edge_power": float(candidate["edge_power"]),
        "candidate_regime_power": float(candidate["regime_power"]),
        "candidate_break_penalty": float(candidate["break_penalty"]),
        "candidate_systemic_penalty": float(candidate["systemic_penalty"]),
        "candidate_vol_penalty": float(candidate["vol_penalty"]),
        "candidate_max_position": float(candidate["max_position"]),
        "trading_improved": bool(model_sort_key(candidate) > model_sort_key(baseline)),
    }


def aggregate_summary(frame: pd.DataFrame) -> dict:
    tickers = int(len(frame))
    trading_improved_count = int(frame["trading_improved"].sum())
    improved_rate = trading_improved_count / tickers if tickers else 0.0
    avg_delta_sharpe = float(frame["delta_sharpe"].mean())
    avg_delta_total_return = float(frame["delta_total_return"].mean())

    return {
        "tickers": tickers,
        "trading_improved_count": trading_improved_count,
        "trading_improved_rate": float(improved_rate),
        "avg_baseline_sharpe": float(frame["baseline_sharpe"].mean()),
        "avg_candidate_sharpe": float(frame["candidate_sharpe"].mean()),
        "avg_delta_sharpe": avg_delta_sharpe,
        "avg_baseline_total_return": float(frame["baseline_total_return"].mean()),
        "avg_candidate_total_return": float(frame["candidate_total_return"].mean()),
        "avg_delta_total_return": avg_delta_total_return,
        "avg_delta_max_drawdown": float(frame["delta_max_drawdown"].mean()),
        "avg_delta_win_rate": float(frame["delta_win_rate"].mean()),
        "avg_delta_f1": float(frame["delta_f1"].mean()),
        "avg_delta_accuracy": float(frame["delta_accuracy"].mean()),
        "avg_candidate_position": float(frame["candidate_avg_position"].mean()),
        "substantial_improvement": bool(
            avg_delta_sharpe >= SUBSTANTIAL_SHARPE_DELTA
            and improved_rate >= SUBSTANTIAL_IMPROVED_RATE
            and avg_delta_total_return > 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prototype factor-residual changepoint fractional sizing.",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Ticker to benchmark. Pass multiple times to limit the run.",
    )
    args = parser.parse_args()

    repo = MarketDataRepository()
    universe_tickers = [
        ticker
        for ticker in repo.list_tickers()
        if not ticker.startswith("^")
    ]
    tickers = args.tickers or [
        ticker
        for ticker in sorted(
            path.stem.replace("_meta", "").replace("_", ".")
            for path in ARTIFACTS_DIR.glob("*_meta.json")
        )
        if not ticker.startswith("^")
    ]

    feature_frames = prepare_universe_feature_frames(repo, universe_tickers)

    rows = []

    for ticker in tickers:
        result = benchmark_ticker(ticker, feature_frames)
        if result is not None:
            rows.append(result)

    frame = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    summary = aggregate_summary(frame)

    frame.to_csv(OUTPUT_DIR / "factor_changepoint_sizing_results.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
