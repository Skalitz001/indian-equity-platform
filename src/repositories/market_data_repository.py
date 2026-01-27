from pathlib import Path
import pandas as pd
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class MarketDataRepository:
    def __init__(self, base_dir: str = "data/raw"):
        self.base_dir = Path(base_dir)

    def _path_for(self, ticker: str) -> Path:
        return self.base_dir / f"{ticker}.csv"

    def list_tickers(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        return [p.stem for p in self.base_dir.glob("*.csv")]

    def load(self, ticker: str) -> pd.DataFrame:
        path = self._path_for(ticker)

        if not path.exists():
            raise FileNotFoundError(f"No data found for ticker: {ticker}")

        logger.info(f"Loading data for {ticker}")
        df = pd.read_csv(path, parse_dates=["Date"])

        
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Optional sanity check
        if df[numeric_cols].isnull().any().any():
            logger.warning(f"NaNs detected after type coercion for {ticker}")

        return df
