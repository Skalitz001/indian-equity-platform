import numpy as np

from src.utils.config import load_config
from src.repositories.market_data_repository import MarketDataRepository
from src.analytics.experimental_features import (
    DEFAULT_CLASSIFICATION_FEATURE_GROUP,
    build_main_feature_frame,
    resolve_main_feature_columns,
)
from src.models.logistic_model import DirectionLogisticModel
from src.models.persistence import save_model

FEATURES = resolve_main_feature_columns(
    DEFAULT_CLASSIFICATION_FEATURE_GROUP
)

def run():

    config = load_config("configs/data.yaml")
    repo = MarketDataRepository()

    for ticker in config["tickers"]:

        print(f"\nTraining model for {ticker}")

        df = repo.load(ticker)

        df = build_main_feature_frame(
            df,
            feature_group=DEFAULT_CLASSIFICATION_FEATURE_GROUP,
        )

        next_log_return = df["log_return"].shift(-1)
        df["target"] = np.where(
            next_log_return.notna(),
            (next_log_return > 0).astype(int),
            np.nan,
        )
        df = df.dropna().reset_index(drop=True)

        X = df[FEATURES]
        y = df["target"]

        model = DirectionLogisticModel()
        model.train(X, y)

        save_model(model, ticker)


if __name__ == "__main__":
    run()
