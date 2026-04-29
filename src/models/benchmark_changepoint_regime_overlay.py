import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.factor_changepoint_features import FACTOR_PREFIX
from src.analytics.performance import max_drawdown, sharpe_ratio, total_return, win_rate
from src.models.benchmark_cross_sectional_portfolio import TRANSACTION_COST


INPUT_PANEL = Path("artifacts/cross_sectional_portfolio_benchmark/candidate_probability_panel.csv")
OUTPUT_DIR = Path("artifacts/changepoint_regime_overlay_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBSTANTIAL_SHARPE_DELTA = 0.003
TRAIN_FRACTION = 0.70

RISK_COLUMNS = {
    "regime_confidence": f"{FACTOR_PREFIX}_regime_confidence",
    "unexplained_break": f"{FACTOR_PREFIX}_unexplained_break",
    "systemic_pressure": f"{FACTOR_PREFIX}_breadth_weighted_systemic_pressure",
    "resid_vol_ratio": f"{FACTOR_PREFIX}_resid_vol_ratio_20_60",
    "factor_cp_breadth": f"{FACTOR_PREFIX}_factor_cp_breadth",
}

LOW_EXPOSURE_GRID = (0.0, 0.25, 0.50, 0.75)
RISK_QUANTILE_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)
SOFT_STRENGTH_GRID = (0.25, 0.50, 0.75, 1.00)
CONFIDENCE_FLOOR_GRID = (0.0, 0.25, 0.50)


def load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["Date"])
    panel["Date"] = panel["Date"].dt.normalize()
    required = ["Date", "ticker", "next_log_return", *RISK_COLUMNS.values()]
    missing = [column for column in required if column not in panel.columns]

    if missing:
        raise ValueError(f"Missing required panel columns: {missing}")

    return panel.dropna(subset=required).sort_values(["Date", "ticker"])


def daily_risk_frame(panel: pd.DataFrame) -> pd.DataFrame:
    daily = panel.groupby("Date").agg({
        RISK_COLUMNS["regime_confidence"]: "mean",
        RISK_COLUMNS["unexplained_break"]: "mean",
        RISK_COLUMNS["systemic_pressure"]: "mean",
        RISK_COLUMNS["resid_vol_ratio"]: "mean",
        RISK_COLUMNS["factor_cp_breadth"]: "mean",
    })
    daily = daily.rename(columns={
        RISK_COLUMNS["regime_confidence"]: "regime_confidence",
        RISK_COLUMNS["unexplained_break"]: "unexplained_break",
        RISK_COLUMNS["systemic_pressure"]: "systemic_pressure",
        RISK_COLUMNS["resid_vol_ratio"]: "resid_vol_ratio",
        RISK_COLUMNS["factor_cp_breadth"]: "factor_cp_breadth",
    })
    daily["resid_vol_excess"] = ((daily["resid_vol_ratio"] - 1.0) / 1.5).clip(0.0, 1.0)
    daily["risk_score"] = (
        0.35 * (1.0 - daily["regime_confidence"].clip(0.0, 1.0))
        + 0.25 * daily["systemic_pressure"].clip(0.0, 1.0)
        + 0.20 * daily["factor_cp_breadth"].clip(0.0, 1.0)
        + 0.15 * daily["unexplained_break"].clip(0.0, 1.0)
        + 0.05 * daily["resid_vol_excess"]
    )

    return daily


def equal_weight_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    returns = panel.pivot(
        index="Date",
        columns="ticker",
        values="next_log_return",
    ).sort_index()
    available = returns.notna().astype(float)

    return available.div(
        available.sum(axis=1).replace(0, np.nan),
        axis=0,
    ).fillna(0.0)


def portfolio_returns(
    returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    exposure: pd.Series,
) -> pd.Series:
    exposure = exposure.reindex(base_weights.index).astype(float).fillna(0.0)
    weights = base_weights.mul(exposure, axis=0)
    turnover = weights.diff().abs().sum(axis=1)

    if len(turnover) > 0:
        turnover.iloc[0] = weights.iloc[0].abs().sum()

    return (weights * returns.fillna(0.0)).sum(axis=1) - (TRANSACTION_COST * turnover)


def metrics(returns: pd.Series) -> dict:
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


def exposure_for_policy(
    risk: pd.DataFrame,
    policy: dict,
    train_risk: pd.DataFrame,
) -> pd.Series:
    kind = policy["kind"]

    if kind == "fully_invested":
        return pd.Series(1.0, index=risk.index)

    if kind == "hard_risk_cut":
        cutoff = train_risk["risk_score"].quantile(policy["risk_quantile"])
        return pd.Series(
            np.where(
                risk["risk_score"] <= cutoff,
                1.0,
                policy["low_exposure"],
            ),
            index=risk.index,
            dtype=float,
        )

    if kind == "soft_risk_scale":
        low = float(train_risk["risk_score"].quantile(0.05))
        high = float(train_risk["risk_score"].quantile(0.95))
        denominator = max(high - low, 1e-9)
        risk_rank = ((risk["risk_score"] - low) / denominator).clip(0.0, 1.0)

        return (1.0 - (policy["strength"] * risk_rank)).clip(
            policy["low_exposure"],
            1.0,
        )

    if kind == "confidence_scale":
        confidence = risk["regime_confidence"].clip(0.0, 1.0)

        return policy["floor"] + ((1.0 - policy["floor"]) * confidence)

    if kind == "confidence_with_systemic_cut":
        confidence = risk["regime_confidence"].clip(0.0, 1.0)
        systemic_cutoff = train_risk["systemic_pressure"].quantile(
            policy["risk_quantile"],
        )
        base = policy["floor"] + ((1.0 - policy["floor"]) * confidence)

        return pd.Series(
            np.where(
                risk["systemic_pressure"] <= systemic_cutoff,
                base,
                np.minimum(base, policy["low_exposure"]),
            ),
            index=risk.index,
            dtype=float,
        )

    raise ValueError(f"Unsupported policy kind: {kind}")


def candidate_policies() -> list[dict]:
    policies = [{"kind": "fully_invested"}]

    for risk_quantile in RISK_QUANTILE_GRID:
        for low_exposure in LOW_EXPOSURE_GRID:
            policies.append({
                "kind": "hard_risk_cut",
                "risk_quantile": float(risk_quantile),
                "low_exposure": float(low_exposure),
            })

    for strength in SOFT_STRENGTH_GRID:
        for low_exposure in LOW_EXPOSURE_GRID:
            policies.append({
                "kind": "soft_risk_scale",
                "strength": float(strength),
                "low_exposure": float(low_exposure),
            })

    for floor in CONFIDENCE_FLOOR_GRID:
        policies.append({
            "kind": "confidence_scale",
            "floor": float(floor),
        })

    for risk_quantile in RISK_QUANTILE_GRID:
        for low_exposure in LOW_EXPOSURE_GRID:
            for floor in CONFIDENCE_FLOOR_GRID:
                policies.append({
                    "kind": "confidence_with_systemic_cut",
                    "risk_quantile": float(risk_quantile),
                    "low_exposure": float(low_exposure),
                    "floor": float(floor),
                })

    return policies


def sort_key(result: dict) -> tuple:
    return (
        result["train"]["sharpe"],
        result["train"]["total_return"],
        result["train"]["max_drawdown"],
        result["train"]["win_rate"],
    )


def evaluate_policy(
    policy: dict,
    risk: pd.DataFrame,
    train_risk: pd.DataFrame,
    returns: pd.DataFrame,
    base_weights: pd.DataFrame,
    train_dates: pd.Index,
    holdout_dates: pd.Index,
) -> dict:
    exposure = exposure_for_policy(risk, policy, train_risk)
    strategy_returns = portfolio_returns(returns, base_weights, exposure)

    return {
        "policy": policy,
        "train": metrics(strategy_returns.loc[train_dates]),
        "holdout": metrics(strategy_returns.loc[holdout_dates]),
        "full": metrics(strategy_returns),
        "avg_exposure_train": float(exposure.loc[train_dates].mean()),
        "avg_exposure_holdout": float(exposure.loc[holdout_dates].mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a changepoint regime-risk exposure overlay.",
    )
    parser.add_argument(
        "--panel",
        default=str(INPUT_PANEL),
        help="Candidate probability panel from the cross-sectional benchmark.",
    )
    args = parser.parse_args()

    panel_path = Path(args.panel)
    panel = load_panel(panel_path)
    risk = daily_risk_frame(panel)
    returns = panel.pivot(
        index="Date",
        columns="ticker",
        values="next_log_return",
    ).sort_index()
    base_weights = equal_weight_matrix(panel).reindex(returns.index).fillna(0.0)

    dates = returns.index.intersection(risk.index)
    split_index = int(len(dates) * TRAIN_FRACTION)
    train_dates = dates[:split_index]
    holdout_dates = dates[split_index:]
    train_risk = risk.loc[train_dates]

    baseline_policy = {"kind": "fully_invested"}
    baseline = evaluate_policy(
        baseline_policy,
        risk,
        train_risk,
        returns,
        base_weights,
        train_dates,
        holdout_dates,
    )
    candidates = [
        evaluate_policy(
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
    best = max(candidates, key=sort_key)

    summary = {
        "panel": str(panel_path),
        "ticker_count": int(panel["ticker"].nunique()),
        "transaction_cost": TRANSACTION_COST,
        "train_fraction": TRAIN_FRACTION,
        "train_start": str(pd.Timestamp(train_dates.min()).date()),
        "train_end": str(pd.Timestamp(train_dates.max()).date()),
        "holdout_start": str(pd.Timestamp(holdout_dates.min()).date()),
        "holdout_end": str(pd.Timestamp(holdout_dates.max()).date()),
        "baseline": baseline,
        "candidate": best,
        "holdout_delta": {
            "sharpe": float(best["holdout"]["sharpe"] - baseline["holdout"]["sharpe"]),
            "total_return": float(
                best["holdout"]["total_return"] - baseline["holdout"]["total_return"]
            ),
            "max_drawdown": float(
                best["holdout"]["max_drawdown"] - baseline["holdout"]["max_drawdown"]
            ),
            "win_rate": float(best["holdout"]["win_rate"] - baseline["holdout"]["win_rate"]),
        },
        "full_delta": {
            "sharpe": float(best["full"]["sharpe"] - baseline["full"]["sharpe"]),
            "total_return": float(best["full"]["total_return"] - baseline["full"]["total_return"]),
            "max_drawdown": float(best["full"]["max_drawdown"] - baseline["full"]["max_drawdown"]),
            "win_rate": float(best["full"]["win_rate"] - baseline["full"]["win_rate"]),
        },
    }
    summary["substantial_holdout_improvement"] = bool(
        summary["holdout_delta"]["sharpe"] >= SUBSTANTIAL_SHARPE_DELTA
        and summary["holdout_delta"]["total_return"] > 0.0
    )

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
