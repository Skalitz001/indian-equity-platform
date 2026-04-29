import numpy as np
import pandas as pd

from src.analytics.changepoint_features import (
    CHANGEPOINT_FEATURE_COLUMNS,
    CHANGEPOINT_SCORE_COLUMNS,
    add_changepoint_features,
    changepoint_scores,
)


FACTOR_PREFIX = "frcp"
FACTOR_RETURN_COLUMN = f"{FACTOR_PREFIX}_factor_log_return"

FACTOR_CONTEXT_COLUMNS = [
    FACTOR_RETURN_COLUMN,
    f"{FACTOR_PREFIX}_factor_abs_return_mean",
    f"{FACTOR_PREFIX}_factor_dispersion",
    f"{FACTOR_PREFIX}_factor_up_fraction",
    f"{FACTOR_PREFIX}_factor_down_fraction",
    f"{FACTOR_PREFIX}_factor_available_count",
    f"{FACTOR_PREFIX}_factor_cp_breadth",
    f"{FACTOR_PREFIX}_factor_var_breadth",
    f"{FACTOR_PREFIX}_factor_dist_breadth",
    f"{FACTOR_PREFIX}_factor_instability_mean",
    f"{FACTOR_PREFIX}_factor_instability_max",
]

FACTOR_TIMESERIES_CHANGEPOINT_COLUMNS = [
    f"{FACTOR_PREFIX}_factor_{column}"
    for column in CHANGEPOINT_SCORE_COLUMNS
]

RESIDUAL_CHANGEPOINT_COLUMNS = [
    f"{FACTOR_PREFIX}_resid_{column}"
    for column in CHANGEPOINT_SCORE_COLUMNS
]

FACTOR_MODEL_COLUMNS = [
    f"{FACTOR_PREFIX}_beta_20",
    f"{FACTOR_PREFIX}_beta_60",
    f"{FACTOR_PREFIX}_beta_shift_20_60",
    f"{FACTOR_PREFIX}_corr_20",
    f"{FACTOR_PREFIX}_corr_60",
    f"{FACTOR_PREFIX}_corr_shift_20_60",
    f"{FACTOR_PREFIX}_r2_20",
    f"{FACTOR_PREFIX}_r2_60",
    f"{FACTOR_PREFIX}_r2_shift_20_60",
    f"{FACTOR_PREFIX}_alpha_60",
    f"{FACTOR_PREFIX}_expected_return_60",
    f"{FACTOR_PREFIX}_residual_return",
    f"{FACTOR_PREFIX}_resid_vol_20",
    f"{FACTOR_PREFIX}_resid_vol_60",
    f"{FACTOR_PREFIX}_resid_vol_ratio_20_60",
    f"{FACTOR_PREFIX}_resid_to_stock_vol_20",
    f"{FACTOR_PREFIX}_resid_abs_z_60",
]

FACTOR_INTERACTION_COLUMNS = [
    f"{FACTOR_PREFIX}_resid_minus_factor_instability",
    f"{FACTOR_PREFIX}_stock_minus_factor_instability",
    f"{FACTOR_PREFIX}_factor_absorption",
    f"{FACTOR_PREFIX}_idio_break_pressure",
    f"{FACTOR_PREFIX}_systemic_break_pressure",
    f"{FACTOR_PREFIX}_breadth_weighted_systemic_pressure",
    f"{FACTOR_PREFIX}_beta_break_pressure",
    f"{FACTOR_PREFIX}_regime_confidence",
    f"{FACTOR_PREFIX}_unexplained_break",
    f"{FACTOR_PREFIX}_explained_break",
]

FACTOR_CHANGEPOINT_FEATURE_COLUMNS = [
    *FACTOR_CONTEXT_COLUMNS,
    *FACTOR_TIMESERIES_CHANGEPOINT_COLUMNS,
    *FACTOR_MODEL_COLUMNS,
    *RESIDUAL_CHANGEPOINT_COLUMNS,
    *FACTOR_INTERACTION_COLUMNS,
]


def _feature_frame_for_cross_section(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    if "cp_instability_score" not in output.columns:
        output = add_changepoint_features(output)

    return output[
        [
            "Date",
            "log_return",
            "cp_any_event_5",
            "cp_var_event",
            "cp_dist_event",
            "cp_instability_score",
        ]
    ].copy()


def build_cross_sectional_factor_frame(
    feature_frames: dict[str, pd.DataFrame],
    exclude_ticker: str,
) -> pd.DataFrame:
    return_columns = {}
    any_event_columns = {}
    var_event_columns = {}
    dist_event_columns = {}
    instability_columns = {}

    for ticker, frame in feature_frames.items():
        if ticker == exclude_ticker or ticker.startswith("^"):
            continue

        prepared = _feature_frame_for_cross_section(frame)
        prepared = prepared.drop_duplicates(subset=["Date"], keep="last")
        prepared = prepared.set_index("Date").sort_index()

        return_columns[ticker] = prepared["log_return"]
        any_event_columns[ticker] = prepared["cp_any_event_5"]
        var_event_columns[ticker] = prepared["cp_var_event"]
        dist_event_columns[ticker] = prepared["cp_dist_event"]
        instability_columns[ticker] = prepared["cp_instability_score"]

    if not return_columns:
        raise ValueError("cross-sectional factor requires at least one peer ticker")

    returns = pd.DataFrame(return_columns).sort_index()
    any_events = pd.DataFrame(any_event_columns).reindex(returns.index)
    var_events = pd.DataFrame(var_event_columns).reindex(returns.index)
    dist_events = pd.DataFrame(dist_event_columns).reindex(returns.index)
    instability = pd.DataFrame(instability_columns).reindex(returns.index)

    context = pd.DataFrame(index=returns.index)
    context[FACTOR_RETURN_COLUMN] = returns.mean(axis=1, skipna=True)
    context[f"{FACTOR_PREFIX}_factor_abs_return_mean"] = returns.abs().mean(
        axis=1,
        skipna=True,
    )
    context[f"{FACTOR_PREFIX}_factor_dispersion"] = returns.std(axis=1, skipna=True)
    context[f"{FACTOR_PREFIX}_factor_up_fraction"] = (
        (returns > 0).sum(axis=1) / returns.notna().sum(axis=1).replace(0, np.nan)
    )
    context[f"{FACTOR_PREFIX}_factor_down_fraction"] = (
        (returns < 0).sum(axis=1) / returns.notna().sum(axis=1).replace(0, np.nan)
    )
    context[f"{FACTOR_PREFIX}_factor_available_count"] = returns.notna().sum(axis=1)
    context[f"{FACTOR_PREFIX}_factor_cp_breadth"] = any_events.mean(
        axis=1,
        skipna=True,
    )
    context[f"{FACTOR_PREFIX}_factor_var_breadth"] = var_events.mean(
        axis=1,
        skipna=True,
    )
    context[f"{FACTOR_PREFIX}_factor_dist_breadth"] = dist_events.mean(
        axis=1,
        skipna=True,
    )
    context[f"{FACTOR_PREFIX}_factor_instability_mean"] = instability.mean(
        axis=1,
        skipna=True,
    )
    context[f"{FACTOR_PREFIX}_factor_instability_max"] = instability.max(
        axis=1,
        skipna=True,
    )

    factor_scores = changepoint_scores(
        context[FACTOR_RETURN_COLUMN],
        prefix=f"{FACTOR_PREFIX}_factor",
    )
    context = context.join(factor_scores)
    context = context.reset_index().rename(columns={"index": "Date"})

    return context.replace([np.inf, -np.inf], np.nan)


def _rolling_beta(
    stock_return: pd.Series,
    factor_return: pd.Series,
    window: int,
) -> pd.Series:
    covariance = stock_return.rolling(window).cov(factor_return)
    variance = factor_return.rolling(window).var().replace(0, np.nan)

    return covariance / variance


def _rolling_corr(
    stock_return: pd.Series,
    factor_return: pd.Series,
    window: int,
) -> pd.Series:
    return stock_return.rolling(window).corr(factor_return)


def add_factor_residual_changepoint_features(
    frame: pd.DataFrame,
    factor_frame: pd.DataFrame,
) -> pd.DataFrame:
    output = frame.copy()

    if "cp_instability_score" not in output.columns:
        output = add_changepoint_features(output)

    output["Date"] = pd.to_datetime(output["Date"], errors="coerce").dt.normalize()
    factor_context = factor_frame.copy()
    factor_context["Date"] = pd.to_datetime(
        factor_context["Date"],
        errors="coerce",
    ).dt.normalize()

    output = output.merge(
        factor_context,
        on="Date",
        how="left",
    )

    stock_return = output["log_return"].astype(float)
    factor_return = output[FACTOR_RETURN_COLUMN].astype(float)

    beta_20 = _rolling_beta(stock_return, factor_return, 20)
    beta_60 = _rolling_beta(stock_return, factor_return, 60)
    corr_20 = _rolling_corr(stock_return, factor_return, 20)
    corr_60 = _rolling_corr(stock_return, factor_return, 60)

    factor_mean_60 = factor_return.rolling(60).mean()
    stock_mean_60 = stock_return.rolling(60).mean()
    alpha_60 = stock_mean_60 - (beta_60 * factor_mean_60)
    expected_return = alpha_60 + (beta_60 * factor_return)
    residual_return = stock_return - expected_return

    resid_vol_20 = residual_return.rolling(20).std()
    resid_vol_60 = residual_return.rolling(60).std()
    stock_vol_20 = stock_return.rolling(20).std()

    output[f"{FACTOR_PREFIX}_beta_20"] = beta_20
    output[f"{FACTOR_PREFIX}_beta_60"] = beta_60
    output[f"{FACTOR_PREFIX}_beta_shift_20_60"] = beta_20 - beta_60
    output[f"{FACTOR_PREFIX}_corr_20"] = corr_20
    output[f"{FACTOR_PREFIX}_corr_60"] = corr_60
    output[f"{FACTOR_PREFIX}_corr_shift_20_60"] = corr_20 - corr_60
    output[f"{FACTOR_PREFIX}_r2_20"] = corr_20.pow(2)
    output[f"{FACTOR_PREFIX}_r2_60"] = corr_60.pow(2)
    output[f"{FACTOR_PREFIX}_r2_shift_20_60"] = (
        output[f"{FACTOR_PREFIX}_r2_20"] - output[f"{FACTOR_PREFIX}_r2_60"]
    )
    output[f"{FACTOR_PREFIX}_alpha_60"] = alpha_60
    output[f"{FACTOR_PREFIX}_expected_return_60"] = expected_return
    output[f"{FACTOR_PREFIX}_residual_return"] = residual_return
    output[f"{FACTOR_PREFIX}_resid_vol_20"] = resid_vol_20
    output[f"{FACTOR_PREFIX}_resid_vol_60"] = resid_vol_60
    output[f"{FACTOR_PREFIX}_resid_vol_ratio_20_60"] = (
        resid_vol_20 / resid_vol_60.replace(0, np.nan)
    )
    output[f"{FACTOR_PREFIX}_resid_to_stock_vol_20"] = (
        resid_vol_20 / stock_vol_20.replace(0, np.nan)
    )
    output[f"{FACTOR_PREFIX}_resid_abs_z_60"] = (
        residual_return.abs() / resid_vol_60.replace(0, np.nan)
    )

    residual_scores = changepoint_scores(
        residual_return,
        prefix=f"{FACTOR_PREFIX}_resid",
    )

    for column in residual_scores.columns:
        output[column] = residual_scores[column]

    factor_instability = output[f"{FACTOR_PREFIX}_factor_instability_score"]
    residual_instability = output[f"{FACTOR_PREFIX}_resid_instability_score"]
    stock_instability = output["cp_instability_score"]
    factor_breadth = output[f"{FACTOR_PREFIX}_factor_cp_breadth"]

    output[f"{FACTOR_PREFIX}_resid_minus_factor_instability"] = (
        residual_instability - factor_instability
    )
    output[f"{FACTOR_PREFIX}_stock_minus_factor_instability"] = (
        stock_instability - factor_instability
    )
    output[f"{FACTOR_PREFIX}_factor_absorption"] = (
        stock_instability - residual_instability
    )
    output[f"{FACTOR_PREFIX}_idio_break_pressure"] = (
        residual_instability * (1.0 - factor_instability.clip(0.0, 1.0))
    )
    output[f"{FACTOR_PREFIX}_systemic_break_pressure"] = (
        stock_instability * factor_instability
    )
    output[f"{FACTOR_PREFIX}_breadth_weighted_systemic_pressure"] = (
        stock_instability * factor_breadth
    )
    output[f"{FACTOR_PREFIX}_beta_break_pressure"] = (
        output[f"{FACTOR_PREFIX}_beta_shift_20_60"].abs()
        * factor_instability
    )
    output[f"{FACTOR_PREFIX}_regime_confidence"] = (
        1.0
        - pd.concat(
            [
                stock_instability,
                factor_instability,
                residual_instability,
                factor_breadth,
            ],
            axis=1,
        ).max(axis=1)
    ).clip(0.0, 1.0)
    output[f"{FACTOR_PREFIX}_unexplained_break"] = (
        residual_instability * stock_instability
    )
    output[f"{FACTOR_PREFIX}_explained_break"] = (
        factor_instability * output[f"{FACTOR_PREFIX}_factor_absorption"].clip(
            lower=0.0,
        )
    )

    return output.replace([np.inf, -np.inf], np.nan)
