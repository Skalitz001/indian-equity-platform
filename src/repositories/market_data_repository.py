from pathlib import Path
import pandas as pd
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class MarketDataRepository:
    """
    Repository for accessing historical market data.
    Abstracts away file-system access.
    """

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
        return pd.read_csv(path, parse_dates=["Date"])
