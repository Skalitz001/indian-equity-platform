from src.utils.config import load_config
from src.repositories.market_data_repository import MarketDataRepository
from src.analytics.features import *
from src.models.logistic_model import DirectionLogisticModel
from src.models.persistence import save_model

FEATURES = [
    "log_return",
    "sma_20",
    "sma_50",
    "rsi",
    "momentum_10",
    "volatility_20",
]


def run():

    config = load_config("configs/data.yaml")
    repo = MarketDataRepository()

    for ticker in config["tickers"]:

        print(f"\nTraining model for {ticker}")

        df = repo.load(ticker)

        df = add_log_returns(df)
        df = add_sma(df, 20)
        df = add_sma(df, 50)
        df = add_rsi(df)
        df = add_momentum(df, 10)
        df = add_volatility(df, 20)

        df = df.dropna()

        df["target"] = (df["log_return"].shift(-1) > 0).astype(int)
        df = df.dropna()

        X = df[FEATURES]
        y = df["target"]

        model = DirectionLogisticModel()
        model.train(X, y)

        save_model(model, ticker)


if __name__ == "__main__":
    run()