import numpy as np
import pandas as pd

from dashboard.pipeline import build_feature_frame
from dashboard.probability_display import prediction_label
from src.gift_nifty.dataset import build_gift_model_frame
from src.models.probabilities import predict_up_probability


def format_display_date(value, fmt="%d %b %Y"):
    if pd.isna(value):
        return "Date unavailable"

    return pd.Timestamp(value).strftime(fmt)


def realized_direction(change_pct):
    if pd.isna(change_pct):
        return "N/A"

    if np.isclose(change_pct, 0.0):
        return "FLAT"

    return "UP" if change_pct > 0 else "DOWN"


def lookup_market_snapshot(history, session_date):
    normalized_date = pd.Timestamp(session_date).normalize()
    matches = history.loc[
        pd.to_datetime(history["Date"]).dt.normalize() == normalized_date
    ]

    if matches.empty:
        return np.nan, np.nan

    row_idx = matches.index[-1]
    close = float(history.loc[row_idx, "Close"])

    if row_idx == 0:
        return close, np.nan

    previous_close = float(history.loc[row_idx - 1, "Close"])

    if previous_close == 0:
        return close, np.nan

    return close, (close / previous_close) - 1


def verification_status(prediction, change_pct, actual_available):
    if not actual_available or prediction == "N/A" or pd.isna(change_pct):
        return "Pending"

    actual_direction = realized_direction(change_pct)

    if actual_direction == "FLAT":
        return "Flat"

    return "Match" if prediction == actual_direction else "Miss"


def projection_inputs(history):
    feature_history = build_feature_frame(history)

    if feature_history.empty:
        return None

    avg_abs_log_return = feature_history["log_return"].abs().tail(20).mean()

    if pd.isna(avg_abs_log_return) or avg_abs_log_return <= 0:
        avg_abs_log_return = 0.01

    avg_range_pct = (
        (history["High"] - history["Low"])
        / history["Close"].replace(0, np.nan)
    ).tail(20).mean()

    if pd.isna(avg_range_pct) or avg_range_pct <= 0:
        avg_range_pct = 0.02

    avg_volume = history["Volume"].tail(20).mean()

    if pd.isna(avg_volume) or avg_volume <= 0:
        avg_volume = float(history["Volume"].iloc[-1])

    return {
        "avg_abs_log_return": float(avg_abs_log_return),
        "avg_range_pct": float(avg_range_pct),
        "avg_volume": float(avg_volume),
    }


def build_projected_row(
    history,
    forecast_date,
    projected_close,
    avg_range_pct,
    avg_volume,
):
    previous_close = float(history["Close"].iloc[-1])
    open_price = previous_close
    range_buffer = max(
        previous_close * avg_range_pct,
        abs(projected_close - open_price),
    )
    high_price = max(open_price, projected_close) + (range_buffer / 2)
    low_price = max(
        0.01,
        min(open_price, projected_close) - (range_buffer / 2),
    )

    return pd.DataFrame(
        [
            {
                "Date": forecast_date,
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": projected_close,
                "Volume": int(max(avg_volume, 1)),
            }
        ]
    )


def history_with_prediction_stub(history, session_date):
    session_ts = pd.Timestamp(session_date).normalize()
    prepared = history.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"]).dt.normalize()
    prepared = prepared.loc[prepared["Date"] <= session_ts].copy()

    if prepared.empty:
        return prepared

    if (prepared["Date"] == session_ts).any():
        return prepared

    previous_close = float(prepared["Close"].iloc[-1])
    avg_range_pct = (
        (prepared["High"] - prepared["Low"])
        / prepared["Close"].replace(0, np.nan)
    ).tail(20).mean()

    if pd.isna(avg_range_pct) or avg_range_pct <= 0:
        avg_range_pct = 0.02

    avg_volume = prepared["Volume"].tail(20).mean()

    if pd.isna(avg_volume) or avg_volume <= 0:
        avg_volume = float(prepared["Volume"].iloc[-1])

    stub = build_projected_row(
        prepared,
        session_ts,
        previous_close,
        float(avg_range_pct),
        float(avg_volume),
    )

    return pd.concat([prepared, stub], ignore_index=True)


def resolve_probability_for_source(
    model_frame,
    source_date,
    model,
    features,
    history,
    feature_group="baseline",
):
    if pd.isna(source_date):
        return np.nan

    normalized_date = pd.Timestamp(source_date).normalize()
    matches = model_frame.loc[
        pd.to_datetime(model_frame["Date"]).dt.normalize() == normalized_date,
        "probability_up",
    ]

    if not matches.empty:
        return float(matches.iloc[-1])

    feature_history = build_feature_frame(
        history,
        feature_group=feature_group,
    )

    if feature_history.empty:
        return np.nan

    missing_features = [
        feature for feature in features
        if feature not in feature_history.columns
    ]

    if missing_features:
        return np.nan

    return float(
        predict_up_probability(
            model,
            feature_history[features].tail(1),
        )[0]
    )


def resolve_gift_probability_for_session(
    model_frame,
    session_date,
    model,
    features,
    history,
    gift_history,
):
    if pd.isna(session_date):
        return np.nan

    normalized_date = pd.Timestamp(session_date).normalize()
    matches = model_frame.loc[
        pd.to_datetime(model_frame["Date"]).dt.normalize() == normalized_date,
        "probability_up",
    ]

    if not matches.empty:
        return float(matches.iloc[-1])

    if history.empty or gift_history.empty:
        return np.nan

    prediction_history = history_with_prediction_stub(
        history,
        normalized_date,
    )

    if prediction_history.empty:
        return np.nan

    feature_history = build_gift_model_frame(
        prediction_history,
        gift_history,
    )

    if feature_history.empty:
        return np.nan

    feature_history = feature_history.loc[
        pd.to_datetime(feature_history["Date"]).dt.normalize() == normalized_date
    ]

    if feature_history.empty:
        return np.nan

    missing_features = [
        feature for feature in features
        if feature not in feature_history.columns
    ]

    if missing_features:
        return np.nan

    return float(
        predict_up_probability(
            model,
            feature_history[features].tail(1),
        )[0]
    )


def build_outlook_row(
    session_name,
    session_date,
    price,
    previous_close,
    probability_up,
    price_basis,
    index_history,
    actual_available,
):
    change_pct = (
        np.nan
        if previous_close in (None, 0)
        else (price / previous_close) - 1
    )
    prediction = (
        prediction_label(float(probability_up))
        if not pd.isna(probability_up) else "N/A"
    )
    index_close, index_change_pct = lookup_market_snapshot(
        index_history,
        session_date,
    )

    return {
        "session": session_name,
        "date": pd.Timestamp(session_date),
        "price": float(price),
        "change_pct": float(change_pct) if not pd.isna(change_pct) else np.nan,
        "prediction": prediction,
        "probability_up": (
            float(probability_up)
            if not pd.isna(probability_up) else np.nan
        ),
        "basis": price_basis,
        "realized_move": (
            realized_direction(change_pct)
            if actual_available else "Pending"
        ),
        "verification": verification_status(
            prediction,
            change_pct,
            actual_available,
        ),
        "index_close": (
            float(index_close)
            if not pd.isna(index_close) else np.nan
        ),
        "index_change_pct": (
            float(index_change_pct)
            if not pd.isna(index_change_pct) else np.nan
        ),
    }


def build_three_day_outlook(
    price_history,
    index_history,
    model_frame,
    model,
    features,
    anchor_date,
    feature_group="baseline",
):
    forecast_history = price_history.copy()
    forecast_history = forecast_history.sort_values("Date").reset_index(drop=True)

    if forecast_history.empty:
        return pd.DataFrame()

    anchor_ts = pd.Timestamp(anchor_date).normalize()
    anchor_matches = forecast_history.index[
        pd.to_datetime(forecast_history["Date"]).dt.normalize() == anchor_ts
    ]

    if len(anchor_matches) == 0:
        return pd.DataFrame()

    anchor_idx = int(anchor_matches[-1])
    outlook_rows = []
    anchor_close = float(forecast_history.loc[anchor_idx, "Close"])
    previous_close = (
        float(forecast_history.loc[anchor_idx - 1, "Close"])
        if anchor_idx > 0 else anchor_close
    )
    prior_session_date = (
        pd.Timestamp(forecast_history.loc[anchor_idx - 1, "Date"])
        if anchor_idx > 0 else pd.NaT
    )
    anchor_history = forecast_history.iloc[:anchor_idx + 1].copy().reset_index(drop=True)

    outlook_rows.append(
        build_outlook_row(
            session_name="Today",
            session_date=forecast_history.loc[anchor_idx, "Date"],
            price=anchor_close,
            previous_close=previous_close,
            probability_up=resolve_probability_for_source(
                model_frame,
                prior_session_date,
                model,
                features,
                anchor_history.iloc[:-1].copy(),
                feature_group=feature_group,
            ),
            price_basis="Actual close from local data",
            index_history=index_history,
            actual_available=True,
        )
    )

    simulated_history = anchor_history

    for step, session_name in enumerate(
        ["Next Trading Day", "Day After"],
        start=1,
    ):
        actual_idx = anchor_idx + step

        if actual_idx < len(forecast_history):
            actual_history = forecast_history.iloc[
                :actual_idx + 1
            ].copy().reset_index(drop=True)
            source_date = pd.Timestamp(
                forecast_history.loc[actual_idx - 1, "Date"]
            )
            actual_close = float(forecast_history.loc[actual_idx, "Close"])
            actual_previous_close = float(
                forecast_history.loc[actual_idx - 1, "Close"]
            )

            outlook_rows.append(
                build_outlook_row(
                    session_name=session_name,
                    session_date=forecast_history.loc[actual_idx, "Date"],
                    price=actual_close,
                    previous_close=actual_previous_close,
                    probability_up=resolve_probability_for_source(
                        model_frame,
                        source_date,
                        model,
                        features,
                        actual_history.iloc[:-1].copy(),
                        feature_group=feature_group,
                    ),
                    price_basis="Actual close from local data",
                    index_history=index_history,
                    actual_available=True,
                )
            )

            simulated_history = actual_history
            continue

        source_date = pd.Timestamp(simulated_history["Date"].iloc[-1])
        probability_up = resolve_probability_for_source(
            model_frame,
            source_date,
            model,
            features,
            simulated_history,
            feature_group=feature_group,
        )
        inputs = projection_inputs(simulated_history)

        if inputs is None:
            break

        base_close = float(simulated_history["Close"].iloc[-1])

        if pd.isna(probability_up):
            break

        direction = prediction_label(probability_up)
        confidence = probability_up if direction == "UP" else 1 - probability_up
        projected_log_return = inputs["avg_abs_log_return"] * (0.75 + confidence)

        if direction == "DOWN":
            projected_log_return *= -1

        projected_close = base_close * np.exp(projected_log_return)
        forecast_date = (
            pd.Timestamp(simulated_history["Date"].iloc[-1])
            + pd.offsets.BDay(1)
        )

        outlook_rows.append(
            build_outlook_row(
                session_name=session_name,
                session_date=forecast_date,
                price=projected_close,
                previous_close=base_close,
                probability_up=probability_up,
                price_basis="Projected close using recent move size",
                index_history=index_history,
                actual_available=False,
            )
        )

        projected_row = build_projected_row(
            simulated_history,
            forecast_date,
            projected_close,
            inputs["avg_range_pct"],
            inputs["avg_volume"],
        )
        simulated_history = pd.concat(
            [simulated_history, projected_row],
            ignore_index=True,
        )

    return pd.DataFrame(outlook_rows)


def build_three_day_outlook_gift(
    price_history,
    gift_history,
    index_history,
    model_frame,
    model,
    features,
    anchor_date,
):
    forecast_history = price_history.copy()
    forecast_history = forecast_history.sort_values("Date").reset_index(drop=True)

    if forecast_history.empty:
        return pd.DataFrame()

    anchor_ts = pd.Timestamp(anchor_date).normalize()
    anchor_matches = forecast_history.index[
        pd.to_datetime(forecast_history["Date"]).dt.normalize() == anchor_ts
    ]

    if len(anchor_matches) == 0:
        return pd.DataFrame()

    gift_max_date = (
        pd.Timestamp(gift_history["Date"].max()).normalize()
        if not gift_history.empty else pd.NaT
    )
    anchor_idx = int(anchor_matches[-1])
    outlook_rows = []
    anchor_close = float(forecast_history.loc[anchor_idx, "Close"])
    previous_close = (
        float(forecast_history.loc[anchor_idx - 1, "Close"])
        if anchor_idx > 0 else anchor_close
    )
    anchor_history = forecast_history.iloc[:anchor_idx + 1].copy().reset_index(drop=True)

    outlook_rows.append(
        build_outlook_row(
            session_name="Today",
            session_date=forecast_history.loc[anchor_idx, "Date"],
            price=anchor_close,
            previous_close=previous_close,
            probability_up=resolve_gift_probability_for_session(
                model_frame,
                forecast_history.loc[anchor_idx, "Date"],
                model,
                features,
                anchor_history.iloc[:-1].copy(),
                gift_history,
            ),
            price_basis="Actual close from local data",
            index_history=index_history,
            actual_available=True,
        )
    )

    simulated_history = anchor_history

    for step, session_name in enumerate(
        ["Next Trading Day", "Day After"],
        start=1,
    ):
        actual_idx = anchor_idx + step

        if actual_idx < len(forecast_history):
            actual_history = forecast_history.iloc[
                :actual_idx + 1
            ].copy().reset_index(drop=True)
            actual_close = float(forecast_history.loc[actual_idx, "Close"])
            actual_previous_close = float(
                forecast_history.loc[actual_idx - 1, "Close"]
            )

            outlook_rows.append(
                build_outlook_row(
                    session_name=session_name,
                    session_date=forecast_history.loc[actual_idx, "Date"],
                    price=actual_close,
                    previous_close=actual_previous_close,
                    probability_up=resolve_gift_probability_for_session(
                        model_frame,
                        forecast_history.loc[actual_idx, "Date"],
                        model,
                        features,
                        actual_history.iloc[:-1].copy(),
                        gift_history,
                    ),
                    price_basis="Actual close from local data",
                    index_history=index_history,
                    actual_available=True,
                )
            )

            simulated_history = actual_history
            continue

        forecast_date = (
            pd.Timestamp(simulated_history["Date"].iloc[-1])
            + pd.offsets.BDay(1)
        )
        probability_up = resolve_gift_probability_for_session(
            model_frame,
            forecast_date,
            model,
            features,
            simulated_history,
            gift_history,
        )
        inputs = projection_inputs(simulated_history)

        if inputs is None:
            break

        base_close = float(simulated_history["Close"].iloc[-1])

        if pd.isna(probability_up):
            break

        direction = prediction_label(probability_up)
        confidence = probability_up if direction == "UP" else 1 - probability_up
        projected_log_return = inputs["avg_abs_log_return"] * (0.75 + confidence)

        if direction == "DOWN":
            projected_log_return *= -1

        projected_close = base_close * np.exp(projected_log_return)
        price_basis = (
            "Projected close using recent move size; "
            "GIFT snapshot available for forecast date"
            if not pd.isna(gift_max_date) and forecast_date <= gift_max_date
            else
            "Projected close using recent move size; "
            "latest GIFT snapshot carried forward"
        )

        outlook_rows.append(
            build_outlook_row(
                session_name=session_name,
                session_date=forecast_date,
                price=projected_close,
                previous_close=base_close,
                probability_up=probability_up,
                price_basis=price_basis,
                index_history=index_history,
                actual_available=False,
            )
        )

        projected_row = build_projected_row(
            simulated_history,
            forecast_date,
            projected_close,
            inputs["avg_range_pct"],
            inputs["avg_volume"],
        )
        simulated_history = pd.concat(
            [simulated_history, projected_row],
            ignore_index=True,
        )

    return pd.DataFrame(outlook_rows)
