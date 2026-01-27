from pathlib import Path
import yfinance as yf
from src.utils.config import load_config
from src.utils.logging import setup_logger


logger = setup_logger(__name__)
RAW_DIR = Path("data/raw")


class NSEDownloader:
    def __init__(self, config_path="configs/data.yaml"):
        self.config = load_config(config_path)["nse"]

    def download_one(self, ticker: str):
        logger.info(f"Downloading {ticker}")
        df = yf.download(
            ticker,
            start=self.config["start_date"],
            end=self.config["end_date"],
            interval=self.config["interval"],
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            logger.warning(f"No data for {ticker}")
            return None
        df.reset_index(inplace=True)
        return df

    def run(self):
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        for ticker in self.config["tickers"]:
            df = self.download_one(ticker)
            if df is not None:
                path = RAW_DIR / f"{ticker}.csv"
                df.to_csv(path, index=False)
                logger.info(f"Saved {ticker} → {path}")


if __name__ == "__main__":
    NSEDownloader().run()
