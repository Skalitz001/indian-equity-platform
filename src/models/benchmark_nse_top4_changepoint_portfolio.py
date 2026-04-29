import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.features import FEATURE_COLUMNS
from src.analytics.performance import max_drawdown, sharpe_ratio, total_return, win_rate
from src.models.benchmark_changepoint_regime_overlay import (
    TRAIN_FRACTION,
    candidate_policies,
    daily_risk_frame,
    equal_weight_matrix as overlay_equal_weight_matrix,
    evaluate_policy as evaluate_overlay_policy,
    sort_key as overlay_sort_key,
)
from src.models.benchmark_cross_sectional_portfolio import (
    CANDIDATE_FEATURE_COLUMNS,
    CANDIDATE_FEATURE_GROUP,
    TRANSACTION_COST,
    build_probability_panel,
    evaluate_portfolio,
    prepare_universe_feature_frames,
    select_portfolio_policy,
)
from src.repositories.market_data_repository import MarketDataRepository


OUTPUT_DIR = Path("artifacts/nse_top4_changepoint_portfolio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NSE_TICKER = "^NSEI"
DEFAULT_TOP_N = 4
DEFAULT_LIQUIDITY_LOOKBACK = 252
SUBSTANTIAL_SHARPE_DELTA = 0.003


def metrics_from_returns(returns: pd.Series) -> dict:
    returns = returns.dropna()
    equity_curve = (1.0 + returns).cumprod()

    return {
        "sharpe": float(sharpe_ratio(returns)),
        "total_return": float(total_return(equity_curve)),
        "max_drawdown": float(max_drawdown(equity_curve)),
        "win_rate": float(win_rate(returns)),
        "observations": int(len(returns)),
        "start_date": str(pd.Timestamp(returns.index.min()).date()),
        "end_date": str(pd.Timestamp(returns.index.max()).date()),
        "avg_daily_return": float(returns.mean()),
        "daily_volatility": float(returns.std()),
    }


def rank_stocks_by_liquidity(
    repo: MarketDataRepository,
    lookback: int,
) -> list[dict]:
    rows = []

    for ticker in repo.list_tickers():
        if ticker.startswith("^"):
            continue

        frame = repo.load(ticker).tail(lookback).copy()
        traded_value = frame["Close"].astype(float) * frame["Volume"].astype(float)
        rows.append({
            "ticker": ticker,
            "median_traded_value": float(traded_value.median()),
            "last_date": str(pd.Timestamp(frame["Date"].max()).date()),
            "rows_used": int(len(frame)),
        })

    return sorted(
        rows,
        key=lambda row: row["median_traded_value"],
        reverse=True,
    )


def split_dates(dates: pd.Index) -> tuple[pd.Index, pd.Index]:
    dates = pd.Index(sorted(pd.to_datetime(dates).normalize().unique()))
    split_index = int(len(dates) * TRAIN_FRACTION)

    return dates[:split_index], dates[split_index:]


def panel_dates(panel: pd.DataFrame) -> pd.Index:
    return pd.Index(pd.to_datetime(panel["Date"]).dt.normalize().unique())


def panel_slice(panel: pd.DataFrame, dates: pd.Index) -> pd.DataFrame:
    return panel.loc[pd.to_datetime(panel["Date"]).dt.normalize().isin(dates)].copy()


def return_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.pivot(
        index="Date",
        columns="ticker",
        values="next_log_return",
    ).sort_index()


def static_weight_returns(
    panel: pd.DataFrame,
    tickers: list[str],
) -> pd.Series:
    returns = return_matrix(panel)
    returns = returns.reindex(columns=tickers)
    available = returns.notna().astype(float)
    weights = available.div(
        available.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1)

    if len(turnover) > 0:
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    return (weights * returns.fillna(0.0)).sum(axis=1) - (
        TRANSACTION_COST * turnover
    )


def evaluate_static_baseline(
    panel: pd.DataFrame,
    tickers: list[str],
    train_dates: pd.Index,
    holdout_dates: pd.Index,
) -> dict:
    returns = static_weight_returns(panel, tickers)

    return {
        "tickers": tickers,
        "train": metrics_from_returns(returns.loc[train_dates]),
        "holdout": metrics_from_returns(returns.loc[holdout_dates]),
        "full": metrics_from_returns(returns.loc[[*train_dates, *holdout_dates]]),
    }


def evaluate_model_policy(
    panel: pd.DataFrame,
    candidate: bool,
    train_dates: pd.Index,
    holdout_dates: pd.Index,
) -> dict:
    train_panel = panel_slice(panel, train_dates)
    holdout_panel = panel_slice(panel, holdout_dates)
    full_panel = panel_slice(panel, pd.Index([*train_dates, *holdout_dates]))
    policy = select_portfolio_policy(
        train_panel,
        candidate=candidate,
    )

    if policy is None:
        raise RuntimeError("Unable to select a portfolio policy")

    return {
        "selected_on": "train",
        "policy": policy,
        "train": evaluate_portfolio(train_panel, policy, candidate=candidate),
        "holdout": evaluate_portfolio(holdout_panel, policy, candidate=candidate),
        "full": evaluate_portfolio(full_panel, policy, candidate=candidate),
    }


def evaluate_regime_overlay(
    candidate_panel: pd.DataFrame,
    train_dates: pd.Index,
    holdout_dates: pd.Index,
) -> dict:
    risk = daily_risk_frame(candidate_panel)
    returns = return_matrix(candidate_panel)
    base_weights = overlay_equal_weight_matrix(candidate_panel).reindex(
        returns.index,
    ).fillna(0.0)
    train_risk = risk.loc[train_dates]
    baseline = evaluate_overlay_policy(
        {"kind": "fully_invested"},
        risk,
        train_risk,
        returns,
        base_weights,
        train_dates,
        holdout_dates,
    )
    candidates = [
        evaluate_overlay_policy(
            policy,
            risk,
            train_risk,
            returns,
            base_weights,
            train_dates,
            holdout_dates,
        )
        for policy in candidate_policies()
    ]
    best = max(candidates, key=overlay_sort_key)

    return {
        "baseline_equal_weight": baseline,
        "candidate": best,
        "holdout_delta": metric_delta(best["holdout"], baseline["holdout"]),
        "full_delta": metric_delta(best["full"], baseline["full"]),
    }


def metric_delta(candidate: dict, baseline: dict) -> dict:
    return {
        "sharpe": float(candidate["sharpe"] - baseline["sharpe"]),
        "total_return": float(candidate["total_return"] - baseline["total_return"]),
        "max_drawdown": float(candidate["max_drawdown"] - baseline["max_drawdown"]),
        "win_rate": float(candidate["win_rate"] - baseline["win_rate"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark NSE plus top-4 local stocks with changepoint features.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of non-index stocks selected by recent traded value.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LIQUIDITY_LOOKBACK,
        help="Rows used for median traded-value ranking.",
    )
    args = parser.parse_args()

    repo = MarketDataRepository()
    liquidity_ranking = rank_stocks_by_liquidity(repo, args.lookback)
    top_stocks = [row["ticker"] for row in liquidity_ranking[: args.top_n]]
    tickers = [NSE_TICKER, *top_stocks]
    feature_frames = prepare_universe_feature_frames(repo, tickers)

    baseline_panel = build_probability_panel(
        tickers,
        feature_frames,
        FEATURE_COLUMNS,
        model_label="baseline_rf",
    )
    candidate_panel = build_probability_panel(
        tickers,
        feature_frames,
        CANDIDATE_FEATURE_COLUMNS,
        model_label=CANDIDATE_FEATURE_GROUP,
    )

    common_dates = panel_dates(baseline_panel).intersection(panel_dates(candidate_panel))
    train_dates, holdout_dates = split_dates(common_dates)

    nse_only = evaluate_static_baseline(
        baseline_panel,
        [NSE_TICKER],
        train_dates,
        holdout_dates,
    )
    equal_weight = evaluate_static_baseline(
        baseline_panel,
        tickers,
        train_dates,
        holdout_dates,
    )
    baseline_model = evaluate_model_policy(
        baseline_panel,
        candidate=False,
        train_dates=train_dates,
        holdout_dates=holdout_dates,
    )
    changepoint_model = evaluate_model_policy(
        candidate_panel,
        candidate=True,
        train_dates=train_dates,
        holdout_dates=holdout_dates,
    )
    regime_overlay = evaluate_regime_overlay(
        candidate_panel,
        train_dates,
        holdout_dates,
    )

    model_holdout_delta = metric_delta(
        changepoint_model["holdout"],
        baseline_model["holdout"],
    )
    model_full_delta = metric_delta(
        changepoint_model["full"],
        baseline_model["full"],
    )
    overlay_holdout_delta = regime_overlay["holdout_delta"]

    summary = {
        "portfolio_name": "nse_plus_top4_changepoint",
        "selection_method": (
            "NSE index plus top non-index tickers ranked by recent "
            "median Close*Volume in the local repository"
        ),
        "nse_ticker": NSE_TICKER,
        "top_n": int(args.top_n),
        "liquidity_lookback_rows": int(args.lookback),
        "liquidity_ranking": liquidity_ranking,
        "tickers": tickers,
        "candidate_feature_group": CANDIDATE_FEATURE_GROUP,
        "train_fraction": TRAIN_FRACTION,
        "train_start": str(pd.Timestamp(train_dates.min()).date()),
        "train_end": str(pd.Timestamp(train_dates.max()).date()),
        "holdout_start": str(pd.Timestamp(holdout_dates.min()).date()),
        "holdout_end": str(pd.Timestamp(holdout_dates.max()).date()),
        "baselines": {
            "nse_only": nse_only,
            "equal_weight_nse_top4": equal_weight,
            "baseline_rf_features": baseline_model,
        },
        "candidates": {
            "changepoint_rf_features": changepoint_model,
            "changepoint_regime_overlay": regime_overlay,
        },
        "deltas": {
            "changepoint_rf_vs_baseline_rf_holdout": model_holdout_delta,
            "changepoint_rf_vs_baseline_rf_full": model_full_delta,
            "regime_overlay_vs_equal_weight_holdout": overlay_holdout_delta,
            "regime_overlay_vs_equal_weight_full": regime_overlay["full_delta"],
        },
    }
    summary["substantial_model_holdout_improvement"] = bool(
        model_holdout_delta["sharpe"] >= SUBSTANTIAL_SHARPE_DELTA
        and model_holdout_delta["total_return"] > 0.0
    )
    summary["substantial_overlay_holdout_improvement"] = bool(
        overlay_holdout_delta["sharpe"] >= SUBSTANTIAL_SHARPE_DELTA
        and overlay_holdout_delta["total_return"] > 0.0
    )

    pd.DataFrame(liquidity_ranking).to_csv(
        OUTPUT_DIR / "liquidity_ranking.csv",
        index=False,
    )
    baseline_panel.to_csv(OUTPUT_DIR / "baseline_probability_panel.csv", index=False)
    candidate_panel.to_csv(OUTPUT_DIR / "candidate_probability_panel.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
