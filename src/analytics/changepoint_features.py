import numpy as np
import pandas as pd

from src.analytics.features import FEATURE_COLUMNS, build_feature_frame


RECENT_WINDOW = 20
PRIOR_WINDOW = 60
MEAN_EVENT_THRESHOLD = 2.0
VAR_EVENT_THRESHOLD = 0.7
DIST_EVENT_THRESHOLD = 1.0
EVENT_DAY_CAP = 252

CHANGEPOINT_FEATURE_COLUMNS = [
    "cp_mean_score_20_60",
    "cp_mean_shift_20_60",
    "cp_var_score_20_60",
    "cp_vol_ratio_20_60",
    "cp_dist_score_20_60",
    "cp_recent_mean_20",
    "cp_prior_mean_60",
    "cp_recent_vol_20",
    "cp_prior_vol_60",
    "cp_mean_event",
    "cp_var_event",
    "cp_dist_event",
    "cp_any_event_5",
    "cp_event_count_20",
    "cp_days_since_mean_event",
    "cp_days_since_var_event",
    "cp_days_since_dist_event",
    "cp_instability_score",
]

PAPER_CHANGEPOINT_FEATURE_COLUMNS = [
    "cp_var_score_20_60",
    "cp_vol_ratio_20_60",
    "cp_dist_score_20_60",
    "cp_recent_vol_20",
    "cp_prior_vol_60",
    "cp_var_event",
    "cp_dist_event",
    "cp_any_event_5",
    "cp_event_count_20",
    "cp_days_since_var_event",
    "cp_days_since_dist_event",
    "cp_instability_score",
]


CHANGEPOINT_SCORE_COLUMNS = [
    "mean_score_20_60",
    "mean_shift_20_60",
    "var_score_20_60",
    "vol_ratio_20_60",
    "dist_score_20_60",
    "recent_mean_20",
    "prior_mean_60",
    "recent_vol_20",
    "prior_vol_60",
    "mean_event",
    "var_event",
    "dist_event",
    "any_event_5",
    "event_count_20",
    "days_since_mean_event",
    "days_since_var_event",
    "days_since_dist_event",
    "instability_score",
]


def _finite_window(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _two_window_changepoint_scores(
    returns: pd.Series,
    recent_window: int = RECENT_WINDOW,
    prior_window: int = PRIOR_WINDOW,
) -> pd.DataFrame:
    values = pd.Series(returns).astype(float).to_numpy()
    length = len(values)

    output = {
        "cp_mean_score_20_60": np.full(length, np.nan),
        "cp_mean_shift_20_60": np.full(length, np.nan),
        "cp_var_score_20_60": np.full(length, np.nan),
        "cp_vol_ratio_20_60": np.full(length, np.nan),
        "cp_dist_score_20_60": np.full(length, np.nan),
        "cp_recent_mean_20": np.full(length, np.nan),
        "cp_prior_mean_60": np.full(length, np.nan),
        "cp_recent_vol_20": np.full(length, np.nan),
        "cp_prior_vol_60": np.full(length, np.nan),
    }

    eps = 1e-12
    min_recent = max(5, int(recent_window * 0.8))
    min_prior = max(20, int(prior_window * 0.8))
    first_index = recent_window + prior_window - 1

    for index in range(first_index, length):
        recent = _finite_window(values[index - recent_window + 1:index + 1])
        prior = _finite_window(
            values[index - recent_window - prior_window + 1:index - recent_window + 1]
        )

        if len(recent) < min_recent or len(prior) < min_prior:
            continue

        recent_mean = float(np.mean(recent))
        prior_mean = float(np.mean(prior))
        recent_var = float(np.var(recent, ddof=1))
        prior_var = float(np.var(prior, ddof=1))

        recent_vol = float(np.sqrt(max(recent_var, 0.0)))
        prior_vol = float(np.sqrt(max(prior_var, 0.0)))
        mean_denom = np.sqrt(
            (recent_var / max(len(recent), 1))
            + (prior_var / max(len(prior), 1))
            + eps
        )

        quantile_grid = (0.10, 0.25, 0.50, 0.75, 0.90)
        recent_quantiles = np.quantile(recent, quantile_grid)
        prior_quantiles = np.quantile(prior, quantile_grid)

        output["cp_mean_score_20_60"][index] = abs(
            recent_mean - prior_mean
        ) / mean_denom
        output["cp_mean_shift_20_60"][index] = recent_mean - prior_mean
        output["cp_var_score_20_60"][index] = abs(
            np.log((recent_var + eps) / (prior_var + eps))
        )
        output["cp_vol_ratio_20_60"][index] = (
            (recent_vol + eps) / (prior_vol + eps)
        )
        output["cp_dist_score_20_60"][index] = float(
            np.mean(np.abs(recent_quantiles - prior_quantiles))
            / (prior_vol + eps)
        )
        output["cp_recent_mean_20"][index] = recent_mean
        output["cp_prior_mean_60"][index] = prior_mean
        output["cp_recent_vol_20"][index] = recent_vol
        output["cp_prior_vol_60"][index] = prior_vol

    return pd.DataFrame(output, index=returns.index)


def _days_since_event(event: pd.Series, cap: int = EVENT_DAY_CAP) -> pd.Series:
    days = []
    current = cap

    for flag in event.fillna(False).astype(bool):
        if flag:
            current = 0
        else:
            current = min(current + 1, cap)

        days.append(current)

    return pd.Series(days, index=event.index, dtype=float)


def changepoint_scores(
    returns: pd.Series,
    prefix: str = "cp",
) -> pd.DataFrame:
    output = _two_window_changepoint_scores(returns)

    output["cp_mean_score_20_60"] = output["cp_mean_score_20_60"].clip(upper=10.0)
    output["cp_var_score_20_60"] = output["cp_var_score_20_60"].clip(upper=10.0)
    output["cp_dist_score_20_60"] = output["cp_dist_score_20_60"].clip(upper=10.0)
    output["cp_vol_ratio_20_60"] = output["cp_vol_ratio_20_60"].clip(
        lower=0.05,
        upper=20.0,
    )

    output["cp_mean_event"] = (
        output["cp_mean_score_20_60"] >= MEAN_EVENT_THRESHOLD
    ).astype(float)
    output["cp_var_event"] = (
        output["cp_var_score_20_60"] >= VAR_EVENT_THRESHOLD
    ).astype(float)
    output["cp_dist_event"] = (
        output["cp_dist_score_20_60"] >= DIST_EVENT_THRESHOLD
    ).astype(float)

    any_event = (
        output[["cp_mean_event", "cp_var_event", "cp_dist_event"]]
        .max(axis=1)
        .astype(float)
    )
    output["cp_any_event_5"] = any_event.rolling(5, min_periods=1).max()
    output["cp_event_count_20"] = any_event.rolling(20, min_periods=1).sum()

    output["cp_days_since_mean_event"] = _days_since_event(output["cp_mean_event"])
    output["cp_days_since_var_event"] = _days_since_event(output["cp_var_event"])
    output["cp_days_since_dist_event"] = _days_since_event(output["cp_dist_event"])

    output["cp_instability_score"] = pd.concat(
        [
            (output["cp_mean_score_20_60"] / 3.0).clip(0.0, 1.0),
            (output["cp_var_score_20_60"] / 1.0).clip(0.0, 1.0),
            (output["cp_dist_score_20_60"] / 1.0).clip(0.0, 1.0),
        ],
        axis=1,
    ).mean(axis=1)

    renamed_columns = {
        f"cp_{column}": f"{prefix}_{column}"
        for column in CHANGEPOINT_SCORE_COLUMNS
    }

    return output.rename(columns=renamed_columns).replace([np.inf, -np.inf], np.nan)


def add_changepoint_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    scores = changepoint_scores(output["log_return"], prefix="cp")

    for column in scores.columns:
        output[column] = scores[column]

    return output.replace([np.inf, -np.inf], np.nan)


def build_changepoint_feature_frame(
    df: pd.DataFrame,
    dropna: bool = True,
) -> pd.DataFrame:
    frame = build_feature_frame(df, dropna=False)
    frame = add_changepoint_features(frame)

    if dropna:
        return frame.dropna(subset=[*FEATURE_COLUMNS, *CHANGEPOINT_FEATURE_COLUMNS])

    return frame
