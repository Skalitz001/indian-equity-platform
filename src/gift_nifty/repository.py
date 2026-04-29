from pathlib import Path

import pandas as pd

from src.gift_nifty.constants import (
    DEFAULT_GIFT_DATA_PATH,
    GIFT_NIFTY_START_DATE,
)


def _find_first_column(columns, aliases):
    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    for alias in aliases:
        match = normalized.get(alias.lower())
        if match is not None:
            return match

    return None


def _parse_numeric(value):
    if pd.isna(value):
        return pd.NA

    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()

    if not text or text in {"-", "--", "null", "None"}:
        return pd.NA

    multiplier = 1.0

    if text.endswith("%"):
        text = text[:-1]

    suffix = text[-1].upper()

    if suffix in {"K", "M", "B"}:
        text = text[:-1]
        multiplier = {
            "K": 1_000.0,
            "M": 1_000_000.0,
            "B": 1_000_000_000.0,
        }[suffix]

    text = text.replace(",", "")

    try:
        return float(text) * multiplier
    except ValueError:
        return pd.NA


def normalize_gift_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(
            columns=["Date", "Open", "High", "Low", "Close", "Volume"]
        )

    date_col = _find_first_column(
        raw_df.columns,
        ["Date", "date", "Time", "time", "trade_date"],
    )
    close_col = _find_first_column(
        raw_df.columns,
        ["Close", "close", "Price", "price", "Last", "last"],
    )
    open_col = _find_first_column(raw_df.columns, ["Open", "open"])
    high_col = _find_first_column(raw_df.columns, ["High", "high"])
    low_col = _find_first_column(raw_df.columns, ["Low", "low"])
    volume_col = _find_first_column(
        raw_df.columns,
        ["Volume", "volume", "Vol.", "vol.", "Vol", "vol"],
    )

    required = {
        "Date": date_col,
        "Open": open_col,
        "High": high_col,
        "Low": low_col,
        "Close": close_col,
    }

    missing = [
        column_name
        for column_name, source in required.items()
        if source is None
    ]

    if missing:
        raise ValueError(
            "gift history is missing required columns: "
            f"{missing}"
        )

    renamed = raw_df.rename(
        columns={
            date_col: "Date",
            open_col: "Open",
            high_col: "High",
            low_col: "Low",
            close_col: "Close",
        }
    ).copy()

    if volume_col is not None:
        renamed = renamed.rename(columns={volume_col: "Volume"})
    else:
        renamed["Volume"] = 0.0

    normalized = renamed[
        ["Date", "Open", "High", "Low", "Close", "Volume"]
    ].copy()

    normalized["Date"] = pd.to_datetime(
        normalized["Date"],
        errors="coerce",
    ).dt.normalize()

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        normalized[column] = normalized[column].map(_parse_numeric)
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        )

    normalized = normalized.dropna(
        subset=["Date", "Open", "High", "Low", "Close"]
    )
    normalized = normalized.sort_values("Date")
    normalized = normalized.drop_duplicates(
        subset=["Date"],
        keep="last",
    ).reset_index(drop=True)

    normalized["Volume"] = normalized["Volume"].fillna(0.0)

    return normalized.loc[
        normalized["Date"] >= GIFT_NIFTY_START_DATE
    ].reset_index(drop=True)


class GiftNiftyRepository:
    def __init__(self, path: str = DEFAULT_GIFT_DATA_PATH):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                "No GIFT Nifty data found. Expected normalized CSV at "
                f"{self.path}"
            )

        raw_df = pd.read_csv(self.path)
        return normalize_gift_history(raw_df)

    def save(self, df: pd.DataFrame) -> Path:
        normalized = normalize_gift_history(df)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(self.path, index=False)
        return self.path
