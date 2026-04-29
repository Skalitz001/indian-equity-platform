import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
from src.utils.config import load_config
from src.utils.logging import setup_logger


logger = setup_logger(__name__)
RAW_DIR = Path("data/raw")


class NSEDownloader:
    def __init__(self, config_path="configs/data.yaml"):
        self.config = load_config(config_path)
        self.raw_dir = Path(self.config.get("data_directory", RAW_DIR))

    def _path_for(self, ticker: str) -> Path:
        return self.raw_dir / f"{ticker}.csv"

    def _resolve_date(self, value):
        if value in (None, "", "null"):
            return None
        return pd.Timestamp(value).normalize()

    def _load_existing(self, ticker: str):
        path = self._path_for(ticker)

        if not path.exists():
            return None

        df = pd.read_csv(path, parse_dates=["Date"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        return df

    def _resolve_download_window(
        self,
        ticker: str,
        start_date=None,
        end_date=None,
        full_refresh: bool = False,
    ):
        configured_start = self._resolve_date(self.config.get("start_date"))
        configured_end = self._resolve_date(self.config.get("end_date"))

        resolved_start = self._resolve_date(start_date) or configured_start
        resolved_end = self._resolve_date(end_date) or configured_end

        if resolved_end is None:
            resolved_end = pd.Timestamp.today().normalize()

        existing = self._load_existing(ticker)
        incremental_update = bool(self.config.get("incremental_update", True))
        overlap_days = int(self.config.get("refresh_overlap_days", 5))

        if (
            not full_refresh
            and start_date is None
            and incremental_update
            and existing is not None
            and not existing.empty
        ):
            latest_local_date = pd.Timestamp(existing["Date"].max()).normalize()
            incremental_start = latest_local_date - pd.Timedelta(days=overlap_days)
            if resolved_start is None:
                resolved_start = incremental_start
            else:
                resolved_start = max(resolved_start, incremental_start)

        if resolved_start is not None and resolved_start > resolved_end:
            resolved_start = resolved_end

        download_end = resolved_end + pd.Timedelta(days=1)

        return (
            resolved_start.strftime("%Y-%m-%d") if resolved_start is not None else None,
            download_end.strftime("%Y-%m-%d"),
            existing,
        )

    def download_one(
        self,
        ticker: str,
        start_date=None,
        end_date=None,
        full_refresh: bool = False,
    ):
        download_start, download_end, existing = self._resolve_download_window(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            full_refresh=full_refresh,
        )

        logger.info(
            f"Downloading {ticker} from {download_start or 'max history'} to {download_end}"
        )
        df = yf.download(
            ticker,
            start=download_start,
            end=download_end,
            interval=self.config["frequency"],
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            logger.warning(f"No data for {ticker}")
            return existing

        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")

        if existing is not None and not full_refresh:
            df = pd.concat([existing, df], ignore_index=True)
            df = df.sort_values("Date")

        df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        return df

    def run(
        self,
        tickers=None,
        start_date=None,
        end_date=None,
        full_refresh: bool = False,
    ):
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        target_tickers = tickers or self.config["tickers"]

        for ticker in target_tickers:
            df = self.download_one(
                ticker,
                start_date=start_date,
                end_date=end_date,
                full_refresh=full_refresh,
            )
            if df is not None:
                path = self._path_for(ticker)
                df.to_csv(path, index=False)
                logger.info(f"Saved {ticker} → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NSE market data")
    parser.add_argument(
        "--ticker",
        action="append",
        dest="tickers",
        help="Ticker to download. Pass multiple times for multiple tickers.",
    )
    parser.add_argument(
        "--start-date",
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Ignore incremental update logic and redownload the selected range.",
    )
    parser.add_argument(
        "--config-path",
        default="configs/data.yaml",
        help="Path to the YAML data config.",
    )

    args = parser.parse_args()

    NSEDownloader(config_path=args.config_path).run(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        full_refresh=args.full_refresh,
    )
