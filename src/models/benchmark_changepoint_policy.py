import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from src.analytics.changepoint_features import (
    CHANGEPOINT_FEATURE_COLUMNS,
    add_changepoint_features,
)
from src.analytics.experimental_features import build_experimental_feature_frame
from src.analytics.features import FEATURE_COLUMNS
from src.analytics.performance import max_drawdown, sharpe_ratio, total_return, win_rate
from src.backtesting.engine import BacktestEngine
from src.models.probabilities import predict_up_probability
from src.models.train_walkforward import (
    model_sort_key,
    select_threshold,
)
from src.repositories.market_data_repository import MarketDataRepository
from src.validation.metrics import classification_metrics_from_probabilities


ARTIFACTS_DIR = Path("artifacts/models")
OUTPUT_DIR = Path("artifacts/changepoint_policy_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSTANTIAL_SHARPE_DELTA = 0.003
SUBSTANTIAL_IMPROVED_RATE = 0.60

POLICY_ENTRY_THRESHOLD_GRID = np.arange(0.50, 0.81, 0.02)
POLICY_EXIT_BUFFER_GRID = (0.00, 0.04, 0.08)
INSTABILITY_CEILINGS = (0.55, 0.65, 0.75)
VAR_EVENT_COOLDOWNS = (0, 3, 5)
DIST_EVENT_COOLDOWNS = (0, 3)
EVENT_COUNT_CEILINGS = (None, 5.0)


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

    return frame.dropna(
        subset=[*FEATURE_COLUMNS, *CHANGEPOINT_FEATURE_COLUMNS, "target"]
    ).reset_index(drop=True)


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


def changepoint_risk_off_mask(
    df: pd.DataFrame,
    policy: dict,
) -> pd.Series:
    mask = pd.Series(False, index=df.index)

    instability_ceiling = policy.get("instability_ceiling")
    if instability_ceiling is not None:
        mask = mask | (
            df["cp_instability_score"].astype(float) > float(instability_ceiling)
        )

    var_cooldown = int(policy.get("var_event_cooldown", 0))
    if var_cooldown > 0:
        mask = mask | (
            df["cp_days_since_var_event"].astype(float) <= float(var_cooldown)
        )

    dist_cooldown = int(policy.get("dist_event_cooldown", 0))
    if dist_cooldown > 0:
        mask = mask | (
            df["cp_days_since_dist_event"].astype(float) <= float(dist_cooldown)
        )

    event_count_ceiling = policy.get("event_count_ceiling")
    if event_count_ceiling is not None:
        mask = mask | (
            df["cp_event_count_20"].astype(float) > float(event_count_ceiling)
        )

    return mask


def generate_changepoint_filtered_signals(
    probabilities: pd.Series,
    df: pd.DataFrame,
    policy: dict,
) -> pd.Series:
    probabilities = pd.Series(probabilities).astype(float)
    aligned_df = df.loc[probabilities.index]
    risk_off = changepoint_risk_off_mask(aligned_df, policy).to_numpy(dtype=bool)
    probability_values = probabilities.to_numpy(dtype=float)

    entry_threshold = float(policy["entry_threshold"])
    exit_threshold = float(policy["exit_threshold"])
    force_flat = bool(policy.get("force_flat", True))

    signals = []
    in_position = 0

    for index, probability in enumerate(probability_values):
        if risk_off[index]:
            if force_flat:
                in_position = 0
            signals.append(in_position)
            continue

        if pd.isna(probability):
            signals.append(in_position)
            continue

        if in_position == 1 and probability < exit_threshold:
            in_position = 0
        elif in_position == 0 and probability > entry_threshold:
            in_position = 1

        signals.append(in_position)

    return pd.Series(signals, index=probabilities.index, dtype=int)


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

    active_days = int(signals.sum())
    entries = int(
        signals.astype(int).diff().clip(lower=0).fillna(signals.iloc[0]).sum()
    )

    return {
        "sharpe": float(sharpe_ratio(returns)),
        "total_return": float(total_return(bt["equity_curve"])),
        "max_drawdown": float(max_drawdown(bt["equity_curve"])),
        "win_rate": float(win_rate(returns)),
        "active_days": active_days,
        "entries": entries,
    }


def changepoint_policy_metrics(
    df: pd.DataFrame,
    probabilities: pd.Series,
    policy: dict,
    engine: BacktestEngine,
) -> dict | None:
    signals = generate_changepoint_filtered_signals(
        probabilities,
        df,
        policy,
    )
    metrics = signal_metrics(df, signals, engine)

    if metrics is None:
        return None

    return {
        **metrics,
        "threshold": float(policy["entry_threshold"]),
        "entry_threshold": float(policy["entry_threshold"]),
        "exit_threshold": float(policy["exit_threshold"]),
        "instability_ceiling": (
            None
            if policy.get("instability_ceiling") is None
            else float(policy["instability_ceiling"])
        ),
        "var_event_cooldown": int(policy.get("var_event_cooldown", 0)),
        "dist_event_cooldown": int(policy.get("dist_event_cooldown", 0)),
        "event_count_ceiling": (
            None
            if policy.get("event_count_ceiling") is None
            else float(policy["event_count_ceiling"])
        ),
    }


def select_changepoint_policy(
    df: pd.DataFrame,
    probabilities: pd.Series,
    engine: BacktestEngine,
) -> dict | None:
    min_active_days = max(20, int(len(probabilities) * 0.02))
    min_entries = max(5, int(len(probabilities) * 0.005))
    best = None

    for entry_threshold in POLICY_ENTRY_THRESHOLD_GRID:
        for exit_buffer in POLICY_EXIT_BUFFER_GRID:
            exit_threshold = max(0.45, entry_threshold - exit_buffer)
            for instability_ceiling in INSTABILITY_CEILINGS:
                for var_event_cooldown in VAR_EVENT_COOLDOWNS:
                    for dist_event_cooldown in DIST_EVENT_COOLDOWNS:
                        for event_count_ceiling in EVENT_COUNT_CEILINGS:
                            policy = {
                                "entry_threshold": float(entry_threshold),
                                "exit_threshold": float(exit_threshold),
                                "instability_ceiling": instability_ceiling,
                                "var_event_cooldown": var_event_cooldown,
                                "dist_event_cooldown": dist_event_cooldown,
                                "event_count_ceiling": event_count_ceiling,
                            }
                            metrics = changepoint_policy_metrics(
                                df,
                                probabilities,
                                policy,
                                engine,
                            )

                            if metrics is None:
                                continue

                            if metrics["active_days"] < min_active_days:
                                continue

                            if metrics["entries"] < min_entries:
                                continue

                            if best is None or model_sort_key(metrics) > model_sort_key(best):
                                best = metrics

    return best


def benchmark_ticker(ticker: str, repo: MarketDataRepository) -> dict | None:
    meta = load_meta(ticker)
    raw_df = repo.load(ticker)
    frame = build_training_frame(raw_df)

    if len(frame) < 250:
        return None

    engine = BacktestEngine(transaction_cost=0.001)
    rf_config = meta["selected_configs"]["rf"]
    probabilities = walk_forward_probabilities(
        frame,
        FEATURE_COLUMNS,
        build_rf_builder(rf_config),
    )

    baseline = select_threshold(
        frame,
        probabilities,
        engine,
    )
    candidate = select_changepoint_policy(
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
        "baseline_total_return": float(baseline["total_return"]),
        "baseline_max_drawdown": float(baseline["max_drawdown"]),
        "baseline_win_rate": float(baseline["win_rate"]),
        "baseline_active_days": int(baseline["active_days"]),
        "baseline_entries": int(baseline["entries"]),
        "baseline_entry_threshold": float(baseline["entry_threshold"]),
        "baseline_exit_threshold": float(baseline["exit_threshold"]),
        "candidate_sharpe": float(candidate["sharpe"]),
        "candidate_total_return": float(candidate["total_return"]),
        "candidate_max_drawdown": float(candidate["max_drawdown"]),
        "candidate_win_rate": float(candidate["win_rate"]),
        "candidate_active_days": int(candidate["active_days"]),
        "candidate_entries": int(candidate["entries"]),
        "candidate_entry_threshold": float(candidate["entry_threshold"]),
        "candidate_exit_threshold": float(candidate["exit_threshold"]),
        "candidate_instability_ceiling": candidate["instability_ceiling"],
        "candidate_var_event_cooldown": int(candidate["var_event_cooldown"]),
        "candidate_dist_event_cooldown": int(candidate["dist_event_cooldown"]),
        "candidate_event_count_ceiling": candidate["event_count_ceiling"],
        "delta_sharpe": float(candidate["sharpe"] - baseline["sharpe"]),
        "delta_total_return": float(candidate["total_return"] - baseline["total_return"]),
        "delta_max_drawdown": float(candidate["max_drawdown"] - baseline["max_drawdown"]),
        "delta_win_rate": float(candidate["win_rate"] - baseline["win_rate"]),
        "delta_active_days": int(candidate["active_days"] - baseline["active_days"]),
        "delta_entries": int(candidate["entries"] - baseline["entries"]),
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
        "avg_delta_active_days": float(frame["delta_active_days"].mean()),
        "avg_delta_entries": float(frame["delta_entries"].mean()),
        "substantial_improvement": bool(
            avg_delta_sharpe >= SUBSTANTIAL_SHARPE_DELTA
            and improved_rate >= SUBSTANTIAL_IMPROVED_RATE
            and avg_delta_total_return > 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prototype changepoint-aware signal policies.",
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

    frame.to_csv(OUTPUT_DIR / "changepoint_policy_results.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
