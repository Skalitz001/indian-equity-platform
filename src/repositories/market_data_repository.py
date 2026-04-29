from pathlib import Path
import pandas as pd
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class MarketDataRepository:
    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)

    def _path_for(self, ticker: str) -> Path:
        return self.base_dir / f"{ticker}.csv"

    def _clean(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if df["Date"].isnull().any():
            dropped_rows = int(df["Date"].isnull().sum())
            logger.warning(
                f"Dropping {dropped_rows} row(s) with missing dates for {ticker}"
            )
            df = df.dropna(subset=["Date"]).reset_index(drop=True)

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        present_numeric_cols = [
            col for col in numeric_cols if col in df.columns
        ]

        for col in present_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if present_numeric_cols and df[present_numeric_cols].isnull().any().any():
            logger.warning(f"NaNs detected after type coercion for {ticker}")

        return df

    def available_date_range(self, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        df = self.load(ticker)

        if df.empty:
            raise ValueError(f"No rows available for ticker: {ticker}")

        return (
            pd.Timestamp(df["Date"].min()).normalize(),
            pd.Timestamp(df["Date"].max()).normalize(),
        )

    @staticmethod
    def slice_between(
        df: pd.DataFrame,
        start_date=None,
        end_date=None,
    ) -> pd.DataFrame:
        result = df.copy()

        if start_date is not None:
            start_ts = pd.Timestamp(start_date).normalize()
            result = result[result["Date"] >= start_ts]

        if end_date is not None:
            end_ts = pd.Timestamp(end_date).normalize()
            result = result[result["Date"] <= end_ts]

        return result.reset_index(drop=True)

    @staticmethod
    def nearest_available_date(
        df: pd.DataFrame,
        target_date,
        direction: str = "backward",
    ):
        if df.empty:
            return pd.NaT

        if direction not in {"backward", "forward"}:
            raise ValueError("direction must be 'backward' or 'forward'")

        target_ts = pd.Timestamp(target_date).normalize()
        dates = pd.to_datetime(df["Date"]).dt.normalize()

        if direction == "backward":
            candidates = dates[dates <= target_ts]
            return pd.Timestamp(candidates.iloc[-1]) if not candidates.empty else pd.NaT

        candidates = dates[dates >= target_ts]
        return pd.Timestamp(candidates.iloc[0]) if not candidates.empty else pd.NaT

    def list_tickers(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return sorted(p.stem for p in self.base_dir.glob("*.csv"))

    def load(self, ticker: str) -> pd.DataFrame:
        path = self._path_for(ticker)

        if not path.exists():
            raise FileNotFoundError(f"No data found for ticker: {ticker}")

        logger.info(f"Loading data for {ticker}")
        df = pd.read_csv(path, parse_dates=["Date"])
        return self._clean(ticker, df)

    def load_between(self, ticker: str, start_date=None, end_date=None) -> pd.DataFrame:
        df = self.load(ticker)
        return self.slice_between(df, start_date=start_date, end_date=end_date)
