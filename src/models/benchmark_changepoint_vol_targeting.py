import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.changepoint_features import (
    CHANGEPOINT_FEATURE_COLUMNS,
    add_changepoint_features,
)
from src.analytics.experimental_features import build_experimental_feature_frame
from src.analytics.features import FEATURE_COLUMNS
from src.analytics.performance import max_drawdown, sharpe_ratio, total_return, win_rate
from src.backtesting.engine import BacktestEngine
from src.models.benchmark_changepoint_features import (
    build_rf_builder,
    walk_forward_probabilities,
)
from src.models.train_walkforward import model_sort_key, select_threshold
from src.repositories.market_data_repository import MarketDataRepository
from src.strategies.signal_policy import generate_probability_signals
from src.validation.metrics import classification_metrics_from_probabilities


ARTIFACTS_DIR = Path("artifacts/models")
OUTPUT_DIR = Path("artifacts/changepoint_vol_targeting_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSTANTIAL_SHARPE_DELTA = 0.003
SUBSTANTIAL_IMPROVED_RATE = 0.60

ENTRY_THRESHOLDS = (0.50, 0.54, 0.58, 0.62)
EXIT_BUFFERS = (0.00, 0.04, 0.08, 0.12)
VOL_POWERS = (0.0, 0.5, 1.0, 1.5)
INSTABILITY_PENALTIES = (0.0, 0.35, 0.70, 1.0)
VAR_COOLDOWNS = (0, 3, 5)
MIN_SCALES = (0.25, 0.50)


def load_meta(ticker: str) -> dict:
    meta_path = ARTIFACTS_DIR / f"{ticker.replace('.','_')}_meta.json"
    return json.loads(meta_path.read_text())


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = build_experimental_feature_frame(df, dropna=False).copy()
    frame = add_changepoint_features(frame)

    next_log_return = frame["log_return"].shift(-1)
    frame["target"] = np.where(
        next_log_return.notna(),
        (next_log_return > 0).astype(int),
        np.nan,
    )

    return frame.dropna(
        subset=[*FEATURE_COLUMNS, *CHANGEPOINT_FEATURE_COLUMNS, "target"]
    ).reset_index(drop=True)


def volatility_targeted_signals(
    df: pd.DataFrame,
    probabilities: pd.Series,
    policy: dict,
) -> pd.Series:
    entry_threshold = float(policy["entry_threshold"])
    exit_threshold = float(policy["exit_threshold"])
    vol_power = float(policy["vol_power"])
    instability_penalty = float(policy["instability_penalty"])
    var_cooldown = int(policy["var_cooldown"])
    min_scale = float(policy["min_scale"])

    base_signals = generate_probability_signals(
        probabilities,
        {
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
        },
    ).astype(float)

    aligned = df.loc[probabilities.index]
    realized_vol = aligned["volatility_20"].astype(float)
    vol_anchor = realized_vol.rolling(252, min_periods=60).median()
    vol_scale = (vol_anchor / realized_vol.replace(0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    vol_scale = vol_scale.fillna(1.0).clip(lower=min_scale, upper=1.0)

    if vol_power != 1.0:
        vol_scale = vol_scale.pow(vol_power)

    instability = aligned["cp_instability_score"].astype(float).fillna(0.0)
    instability_scale = (
        1.0 - (instability_penalty * instability.clip(0.0, 1.0))
    ).clip(lower=min_scale, upper=1.0)

    if var_cooldown > 0:
        recent_var_break = (
            aligned["cp_days_since_var_event"].astype(float) <= float(var_cooldown)
        )
        instability_scale = instability_scale.where(
            ~recent_var_break,
            np.minimum(instability_scale, min_scale),
        )

    positions = base_signals * vol_scale * instability_scale

    return positions.clip(lower=0.0, upper=1.0).fillna(0.0)


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
        signals.gt(0.01).astype(int).diff().clip(lower=0).fillna(
            signals.iloc[0] > 0.01
        ).sum()
    )

    return {
        "sharpe": float(sharpe_ratio(returns)),
        "total_return": float(total_return(bt["equity_curve"])),
        "max_drawdown": float(max_drawdown(bt["equity_curve"])),
        "win_rate": float(win_rate(returns)),
        "active_days": active_days,
        "entries": entries,
        "avg_position": float(signals.mean()),
    }


def select_vol_targeting_policy(
    df: pd.DataFrame,
    probabilities: pd.Series,
    engine: BacktestEngine,
) -> dict | None:
    min_active_days = max(20, int(len(probabilities) * 0.02))
    min_entries = max(5, int(len(probabilities) * 0.005))
    best = None

    for entry_threshold in ENTRY_THRESHOLDS:
        for exit_buffer in EXIT_BUFFERS:
            exit_threshold = max(0.45, entry_threshold - exit_buffer)
            for vol_power in VOL_POWERS:
                for instability_penalty in INSTABILITY_PENALTIES:
                    for var_cooldown in VAR_COOLDOWNS:
                        for min_scale in MIN_SCALES:
                            policy = {
                                "entry_threshold": entry_threshold,
                                "exit_threshold": exit_threshold,
                                "vol_power": vol_power,
                                "instability_penalty": instability_penalty,
                                "var_cooldown": var_cooldown,
                                "min_scale": min_scale,
                            }
                            signals = volatility_targeted_signals(
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
                            }

                            if best is None or model_sort_key(result) > model_sort_key(best):
                                best = result

    return best


def benchmark_ticker(ticker: str, repo: MarketDataRepository) -> dict | None:
    meta = load_meta(ticker)
    raw_df = repo.load(ticker)
    frame = build_training_frame(raw_df)

    if len(frame) < 250:
        return None

    engine = BacktestEngine(transaction_cost=0.001)
    probabilities = walk_forward_probabilities(
        frame,
        FEATURE_COLUMNS,
        build_rf_builder(meta["selected_configs"]["rf"]),
    )

    baseline = select_threshold(
        frame,
        probabilities,
        engine,
    )
    candidate = select_vol_targeting_policy(
        frame,
        probabilities,
        engine,
    )

    if baseline is None or candidate is None:
        return None

    classification = classification_metrics_from_probabilities(
        frame.loc[probabilities.index, "target"],
        probabilities,
        optimize_threshold=True,
    )

    return {
        "ticker": ticker,
        "rows": int(len(frame)),
        "start_date": str(pd.Timestamp(frame["Date"].min()).date()),
        "end_date": str(pd.Timestamp(frame["Date"].max()).date()),
        "classification_f1": float(classification["f1"]),
        "classification_accuracy": float(classification["accuracy"]),
        "baseline_sharpe": float(baseline["sharpe"]),
        "candidate_sharpe": float(candidate["sharpe"]),
        "delta_sharpe": float(candidate["sharpe"] - baseline["sharpe"]),
        "baseline_total_return": float(baseline["total_return"]),
        "candidate_total_return": float(candidate["total_return"]),
        "delta_total_return": float(candidate["total_return"] - baseline["total_return"]),
        "baseline_max_drawdown": float(baseline["max_drawdown"]),
        "candidate_max_drawdown": float(candidate["max_drawdown"]),
        "delta_max_drawdown": float(candidate["max_drawdown"] - baseline["max_drawdown"]),
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
        "candidate_entry_threshold": float(candidate["entry_threshold"]),
        "candidate_exit_threshold": float(candidate["exit_threshold"]),
        "candidate_vol_power": float(candidate["vol_power"]),
        "candidate_instability_penalty": float(candidate["instability_penalty"]),
        "candidate_var_cooldown": int(candidate["var_cooldown"]),
        "candidate_min_scale": float(candidate["min_scale"]),
        "trading_improved": bool(model_sort_key(candidate) > model_sort_key(baseline)),
    }


def aggregate_summary(frame: pd.DataFrame) -> dict:
    tickers = int(len(frame))
    trading_improved_count = int(frame["trading_improved"].sum())
    improved_rate = trading_improved_count / tickers if tickers else 0.0
    avg_delta_sharpe = float(frame["delta_sharpe"].mean())

    return {
        "tickers": tickers,
        "trading_improved_count": trading_improved_count,
        "trading_improved_rate": float(improved_rate),
        "avg_baseline_sharpe": float(frame["baseline_sharpe"].mean()),
        "avg_candidate_sharpe": float(frame["candidate_sharpe"].mean()),
        "avg_delta_sharpe": avg_delta_sharpe,
        "avg_baseline_total_return": float(frame["baseline_total_return"].mean()),
        "avg_candidate_total_return": float(frame["candidate_total_return"].mean()),
        "avg_delta_total_return": float(frame["delta_total_return"].mean()),
        "avg_delta_max_drawdown": float(frame["delta_max_drawdown"].mean()),
        "avg_delta_win_rate": float(frame["delta_win_rate"].mean()),
        "avg_delta_active_days": float(frame["delta_active_days"].mean()),
        "avg_delta_entries": float(frame["delta_entries"].mean()),
        "avg_candidate_position": float(frame["candidate_avg_position"].mean()),
        "substantial_improvement": bool(
            avg_delta_sharpe >= SUBSTANTIAL_SHARPE_DELTA
            and improved_rate >= SUBSTANTIAL_IMPROVED_RATE
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prototype changepoint volatility-targeted position sizing.",
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
        result = benchmark_ticker(ticker, repo)
        if result is not None:
            rows.append(result)

    frame = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    summary = aggregate_summary(frame)

    frame.to_csv(OUTPUT_DIR / "changepoint_vol_targeting_results.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
