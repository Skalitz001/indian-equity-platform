import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.analytics.factor_changepoint_features import FACTOR_PREFIX
from src.analytics.features import FEATURE_COLUMNS
from src.analytics.performance import max_drawdown, sharpe_ratio, total_return, win_rate
from src.models.benchmark_factor_changepoint_features import (
    FACTOR_FEATURE_GROUPS,
    build_rf_builder,
    build_training_frame,
    load_meta,
    prepare_universe_feature_frames,
)
from src.models.train_walkforward import model_sort_key
from src.repositories.market_data_repository import MarketDataRepository


OUTPUT_DIR = Path("artifacts/cross_sectional_portfolio_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSACTION_COST = 0.001
SUBSTANTIAL_SHARPE_DELTA = 0.003
FAST_RF_N_ESTIMATORS = 40
FAST_WALK_FORWARD_SPLITS = 3
FAST_MAX_TRAIN_ROWS = 900

TOP_K_GRID = (1, 2, 3)
MIN_PROBABILITY_GRID = (0.50, 0.53, 0.56)
BASELINE_WEIGHT_SCHEMES = ("equal", "probability_edge", "inverse_vol")
CANDIDATE_SCORE_MODES = (
    "probability",
    "regime_adjusted",
    "idio_penalized",
    "systemic_penalized",
)
CANDIDATE_WEIGHT_SCHEMES = (
    "equal",
    "probability_edge",
    "inverse_vol",
    "regime_confidence",
)
CANDIDATE_FEATURE_GROUP = "factor_residual_context_cp"
CANDIDATE_FEATURE_COLUMNS = [
    *FEATURE_COLUMNS,
    *FACTOR_FEATURE_GROUPS[CANDIDATE_FEATURE_GROUP],
]


def capped_rf_config(config: dict) -> dict:
    capped = dict(config)
    capped["n_estimators"] = min(
        int(capped.get("n_estimators", FAST_RF_N_ESTIMATORS)),
        FAST_RF_N_ESTIMATORS,
    )

    if capped.get("max_depth") is not None:
        capped["max_depth"] = min(int(capped["max_depth"]), 6)

    return capped


def fast_walk_forward_probabilities(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_builder,
) -> pd.Series:
    probabilities = pd.Series(index=df.index, dtype=float)
    n_splits = min(FAST_WALK_FORWARD_SPLITS, max(2, len(df) // 250))
    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        max_train_size=FAST_MAX_TRAIN_ROWS,
    )

    for train_idx, test_idx in splitter.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = model_builder()
        model.fit(
            train_df[feature_columns],
            train_df["target"],
        )
        probabilities.iloc[test_idx] = model.predict_proba(
            test_df[feature_columns],
        )[:, 1]

    return probabilities.dropna()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def build_probability_panel(
    tickers: list[str],
    feature_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    model_label: str,
) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        frame = build_training_frame(ticker, feature_frames)

        if len(frame) < 250:
            continue

        meta = load_meta(ticker)
        rf_config = capped_rf_config(meta["selected_configs"]["rf"])
        probabilities = fast_walk_forward_probabilities(
            frame,
            feature_columns,
            build_rf_builder(rf_config),
        )

        panel = frame.loc[probabilities.index].copy()
        panel["ticker"] = ticker
        panel["model_label"] = model_label
        panel["probability"] = probabilities
        panel["next_log_return"] = panel["log_return"].shift(-1)
        panel["stock_volatility"] = panel["volatility_20"].astype(float)

        required_columns = [
            "Date",
            "ticker",
            "model_label",
            "probability",
            "next_log_return",
            "stock_volatility",
        ]

        optional_columns = [
            f"{FACTOR_PREFIX}_regime_confidence",
            f"{FACTOR_PREFIX}_unexplained_break",
            f"{FACTOR_PREFIX}_breadth_weighted_systemic_pressure",
            f"{FACTOR_PREFIX}_resid_vol_ratio_20_60",
            f"{FACTOR_PREFIX}_factor_cp_breadth",
        ]

        available_optional = [
            column
            for column in optional_columns
            if column in panel.columns
        ]

        rows.append(panel[[*required_columns, *available_optional]])
        print(
            f"{model_label}: built probability panel for {ticker} "
            f"({len(probabilities)} rows)",
            flush=True,
        )

    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce").dt.normalize()
    result = result.dropna(
        subset=[
            "Date",
            "ticker",
            "probability",
            "next_log_return",
        ]
    )

    return result.sort_values(["Date", "ticker"]).reset_index(drop=True)


def add_default_candidate_columns(panel: pd.DataFrame) -> pd.DataFrame:
    result = panel.copy()

    defaults = {
        f"{FACTOR_PREFIX}_regime_confidence": 1.0,
        f"{FACTOR_PREFIX}_unexplained_break": 0.0,
        f"{FACTOR_PREFIX}_breadth_weighted_systemic_pressure": 0.0,
        f"{FACTOR_PREFIX}_resid_vol_ratio_20_60": 1.0,
        f"{FACTOR_PREFIX}_factor_cp_breadth": 0.0,
    }

    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
        result[column] = result[column].astype(float).fillna(default)

    return result


def score_panel(
    panel: pd.DataFrame,
    score_mode: str,
) -> pd.Series:
    probability = panel["probability"].astype(float)

    if score_mode == "probability":
        return probability

    regime_confidence = panel[f"{FACTOR_PREFIX}_regime_confidence"].clip(0.0, 1.0)
    unexplained_break = panel[f"{FACTOR_PREFIX}_unexplained_break"].clip(0.0, 1.0)
    systemic_break = panel[
        f"{FACTOR_PREFIX}_breadth_weighted_systemic_pressure"
    ].clip(0.0, 1.0)

    if score_mode == "regime_adjusted":
        return probability * (0.5 + (0.5 * regime_confidence))

    if score_mode == "idio_penalized":
        return probability - (0.08 * unexplained_break)

    if score_mode == "systemic_penalized":
        return probability - (0.08 * systemic_break)

    raise ValueError(f"Unsupported score mode: {score_mode}")


def build_weight_matrix(
    panel: pd.DataFrame,
    policy: dict,
    candidate: bool,
) -> pd.DataFrame:
    working = add_default_candidate_columns(panel) if candidate else panel.copy()
    score_mode = policy.get("score_mode", "probability")
    working["score"] = score_panel(working, score_mode)

    min_probability = float(policy["min_probability"])
    top_k = int(policy["top_k"])
    weight_scheme = str(policy["weight_scheme"])

    selected = working.loc[
        working["probability"].astype(float) >= min_probability
    ].copy()

    if selected.empty:
        return pd.DataFrame()

    selected = selected.sort_values(
        ["Date", "score", "probability"],
        ascending=[True, False, False],
    )
    selected = selected.groupby("Date", sort=False).head(top_k).copy()

    if weight_scheme == "equal":
        raw_weight = pd.Series(1.0, index=selected.index)
    elif weight_scheme == "probability_edge":
        raw_weight = (selected["probability"] - min_probability).clip(lower=0.001)
    elif weight_scheme == "inverse_vol":
        raw_weight = _safe_divide(
            pd.Series(1.0, index=selected.index),
            selected["stock_volatility"].astype(float),
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif weight_scheme == "regime_confidence":
        raw_weight = selected[f"{FACTOR_PREFIX}_regime_confidence"].clip(0.05, 1.0)
    else:
        raise ValueError(f"Unsupported weight scheme: {weight_scheme}")

    selected["raw_weight"] = raw_weight.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)
    daily_raw_sum = selected.groupby("Date")["raw_weight"].transform("sum")
    selected.loc[daily_raw_sum <= 0, "raw_weight"] = 1.0
    daily_raw_sum = selected.groupby("Date")["raw_weight"].transform("sum")
    selected["weight"] = selected["raw_weight"] / daily_raw_sum

    return selected.pivot(
        index="Date",
        columns="ticker",
        values="weight",
    ).fillna(0.0)


def evaluate_portfolio(
    panel: pd.DataFrame,
    policy: dict,
    candidate: bool,
) -> dict | None:
    weights = build_weight_matrix(panel, policy, candidate=candidate)

    if weights.empty:
        return None

    returns = panel.pivot(
        index="Date",
        columns="ticker",
        values="next_log_return",
    ).reindex(weights.index)
    returns = returns.fillna(0.0)

    weights = weights.reindex(columns=returns.columns, fill_value=0.0)
    turnover = weights.diff().abs().sum(axis=1)
    if len(turnover) > 0:
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    portfolio_returns = (weights * returns).sum(axis=1) - (
        TRANSACTION_COST * turnover
    )
    portfolio_returns = portfolio_returns.dropna()

    if len(portfolio_returns) < 5:
        return None

    equity_curve = (1.0 + portfolio_returns).cumprod()
    active_days = int(weights.sum(axis=1).gt(0).sum())

    result = {
        **policy,
        "sharpe": float(sharpe_ratio(portfolio_returns)),
        "total_return": float(total_return(equity_curve)),
        "max_drawdown": float(max_drawdown(equity_curve)),
        "win_rate": float(win_rate(portfolio_returns)),
        "active_days": active_days,
        "entries": int(weights.gt(0).sum(axis=1).sum()),
        "avg_gross": float(weights.sum(axis=1).mean()),
        "avg_names": float(weights.gt(0).sum(axis=1).mean()),
        "avg_turnover": float(turnover.mean()),
        "start_date": str(pd.Timestamp(portfolio_returns.index.min()).date()),
        "end_date": str(pd.Timestamp(portfolio_returns.index.max()).date()),
        "observations": int(len(portfolio_returns)),
    }

    return result


def portfolio_sort_key(result: dict) -> tuple:
    return (
        result["sharpe"],
        result["total_return"],
        result["max_drawdown"],
        result["win_rate"],
        -result["avg_turnover"],
        result["active_days"],
    )


def select_portfolio_policy(
    panel: pd.DataFrame,
    candidate: bool,
) -> dict | None:
    best = None
    score_modes = CANDIDATE_SCORE_MODES if candidate else ("probability",)
    weight_schemes = (
        CANDIDATE_WEIGHT_SCHEMES
        if candidate
        else BASELINE_WEIGHT_SCHEMES
    )

    for top_k in TOP_K_GRID:
        for min_probability in MIN_PROBABILITY_GRID:
            for score_mode in score_modes:
                for weight_scheme in weight_schemes:
                    policy = {
                        "top_k": int(top_k),
                        "min_probability": float(min_probability),
                        "score_mode": score_mode,
                        "weight_scheme": weight_scheme,
                    }
                    result = evaluate_portfolio(
                        panel,
                        policy,
                        candidate=candidate,
                    )

                    if result is None:
                        continue

                    if best is None or portfolio_sort_key(result) > portfolio_sort_key(best):
                        best = result

    return best


def equal_weight_universe_metrics(panel: pd.DataFrame) -> dict | None:
    returns = panel.pivot(
        index="Date",
        columns="ticker",
        values="next_log_return",
    ).sort_index()

    if returns.empty:
        return None

    available = returns.notna().astype(float)
    weights = available.div(available.sum(axis=1).replace(0, np.nan), axis=0)
    weights = weights.fillna(0.0)
    portfolio_returns = (weights * returns.fillna(0.0)).sum(axis=1)
    portfolio_returns = portfolio_returns.dropna()

    if len(portfolio_returns) < 5:
        return None

    equity_curve = (1.0 + portfolio_returns).cumprod()

    return {
        "sharpe": float(sharpe_ratio(portfolio_returns)),
        "total_return": float(total_return(equity_curve)),
        "max_drawdown": float(max_drawdown(equity_curve)),
        "win_rate": float(win_rate(portfolio_returns)),
        "observations": int(len(portfolio_returns)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark cross-sectional portfolio ranking with factor changepoint features.",
    )
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Ticker to include. Pass multiple times to limit the universe.",
    )
    args = parser.parse_args()

    repo = MarketDataRepository()
    tickers = args.tickers or [
        ticker
        for ticker in repo.list_tickers()
        if not ticker.startswith("^")
    ]
    tickers = sorted(tickers)
    feature_frames = prepare_universe_feature_frames(repo, tickers)

    baseline_panel = build_probability_panel(
        tickers,
        feature_frames,
        FEATURE_COLUMNS,
        model_label="baseline",
    )
    candidate_panel = build_probability_panel(
        tickers,
        feature_frames,
        CANDIDATE_FEATURE_COLUMNS,
        model_label=CANDIDATE_FEATURE_GROUP,
    )

    baseline = select_portfolio_policy(
        baseline_panel,
        candidate=False,
    )
    candidate = select_portfolio_policy(
        candidate_panel,
        candidate=True,
    )
    equal_weight = equal_weight_universe_metrics(baseline_panel)

    if baseline is None or candidate is None:
        raise RuntimeError("Unable to evaluate portfolio policies")

    summary = {
        "tickers": tickers,
        "ticker_count": len(tickers),
        "candidate_feature_group": CANDIDATE_FEATURE_GROUP,
        "rf_n_estimators_cap": FAST_RF_N_ESTIMATORS,
        "walk_forward_splits": FAST_WALK_FORWARD_SPLITS,
        "max_train_rows": FAST_MAX_TRAIN_ROWS,
        "baseline": baseline,
        "candidate": candidate,
        "equal_weight_universe": equal_weight,
        "delta": {
            "sharpe": float(candidate["sharpe"] - baseline["sharpe"]),
            "total_return": float(candidate["total_return"] - baseline["total_return"]),
            "max_drawdown": float(candidate["max_drawdown"] - baseline["max_drawdown"]),
            "win_rate": float(candidate["win_rate"] - baseline["win_rate"]),
            "avg_turnover": float(candidate["avg_turnover"] - baseline["avg_turnover"]),
        },
    }
    summary["substantial_improvement"] = bool(
        summary["delta"]["sharpe"] >= SUBSTANTIAL_SHARPE_DELTA
        and summary["candidate"]["total_return"] > summary["baseline"]["total_return"]
    )

    baseline_panel.to_csv(OUTPUT_DIR / "baseline_probability_panel.csv", index=False)
    candidate_panel.to_csv(OUTPUT_DIR / "candidate_probability_panel.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
