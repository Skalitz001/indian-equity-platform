import argparse

import pandas as pd
import requests

from src.gift_nifty.constants import (
    DEFAULT_GIFT_DATA_PATH,
    GIFT_NIFTY_START_DATE,
)
from src.gift_nifty.repository import GiftNiftyRepository


DHAN_GIFT_HISTORY_URL = "https://openweb-ticks.dhan.co/getDataH"
DHAN_GIFT_SYMBOL = {
    "EXCH": "IDX",
    "SYM": "GIFTNIFTY",
    "SEG": "I",
    "INST": "IDX",
    "EXPCODE": 0,
    "SEC_ID": 5024,
}


def _to_epoch_seconds(value) -> int:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Kolkata")
    else:
        timestamp = timestamp.tz_convert("Asia/Kolkata")

    return int(timestamp.timestamp())


def fetch_gift_nifty_history(
    start_date=GIFT_NIFTY_START_DATE,
    end_date=None,
    timeout: int = 30,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = (
        pd.Timestamp(end_date).normalize()
        if end_date is not None
        else pd.Timestamp.now(tz="Asia/Kolkata").normalize()
    )

    payload = {
        **DHAN_GIFT_SYMBOL,
        "START": _to_epoch_seconds(start_ts),
        "END": _to_epoch_seconds(end_ts + pd.Timedelta(days=1)),
        "INTERVAL": "D",
    }

    response = requests.post(
        DHAN_GIFT_HISTORY_URL,
        json=payload,
        headers={
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0",
            "origin": "https://dhan.co",
            "referer": "https://dhan.co/indices/gift-nifty-historical-price/",
        },
        timeout=timeout,
    )
    response.raise_for_status()

    body = response.json()
    data = body.get("data", {})

    if not data or "Time" not in data:
        raise ValueError(
            "Dhan GIFT history response did not contain the expected OHLC payload"
        )

    frame = pd.DataFrame(
        {
            "Date": data["Time"],
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data.get("v", [0] * len(data["Time"])),
        }
    )

    return frame


def seed_gift_nifty_csv(
    path: str = DEFAULT_GIFT_DATA_PATH,
    start_date=GIFT_NIFTY_START_DATE,
    end_date=None,
) -> pd.DataFrame:
    history = fetch_gift_nifty_history(
        start_date=start_date,
        end_date=end_date,
    )
    repo = GiftNiftyRepository(path)
    repo.save(history)
    return repo.load()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download post-2023 GIFT Nifty OHLC history from Dhan",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_GIFT_DATA_PATH,
        help="Destination CSV path for normalized GIFT Nifty history.",
    )
    parser.add_argument(
        "--start-date",
        default=str(GIFT_NIFTY_START_DATE.date()),
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Inclusive end date in YYYY-MM-DD format.",
    )

    args = parser.parse_args()

    df = seed_gift_nifty_csv(
        path=args.path,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(
        "Saved",
        len(df),
        "rows to",
        args.path,
        "from",
        str(pd.Timestamp(df["Date"].min()).date()) if not df.empty else "N/A",
        "to",
        str(pd.Timestamp(df["Date"].max()).date()) if not df.empty else "N/A",
    )
