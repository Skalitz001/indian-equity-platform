from src.repositories.market_data_repository import MarketDataRepository
from src.analytics.features import add_log_returns, add_sma
from src.models.logistic_model import DirectionLogisticModel
import pandas as pd


FEATURES = ["log_return", "sma_20", "sma_50"]


def main():
    repo = MarketDataRepository()
    df = repo.load("RELIANCE.NS")

    df = add_log_returns(df)
    df = add_sma(df, 20)
    df = add_sma(df, 50)

    df["target"] = (df["log_return"].shift(-1) > 0).astype(int)
    df = df.dropna()

    X = df[FEATURES]
    y = df["target"]

    model = DirectionLogisticModel()
    model.train(X, y)

    print("Model trained successfully")
    return model


if __name__ == "__main__":
    main()
