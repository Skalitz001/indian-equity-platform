import pandas as pd
from typing import List

from src.models.probabilities import predict_up_probability
from src.validation.metrics import (
    find_best_threshold,
    threshold_predictions,
)


class WalkForwardValidator:

    def __init__(

        self,

        model,

        feature_columns: List[str],

        target_column: str,

        train_window: int,

        step_size: int,

        threshold: float = 0.57,

    ):

        self.model=model

        self.feature_columns=feature_columns

        self.target_column=target_column

        self.train_window=train_window

        self.step_size=step_size

        self.threshold=threshold


    def run(self,df):

        signals=pd.Series(
            index=df.index,
            dtype=float
        )

        start=self.train_window

        end=len(df)


        while start<end:

            train_start=start-self.train_window

            train_end=start

            test_end=min(

                start+self.step_size,

                end

            )


            train_df=df.iloc[
                train_start:train_end
            ]

            test_df=df.iloc[
                train_end:test_end
            ]


            X_train=train_df[
                self.feature_columns
            ]

            y_train=train_df[
                self.target_column
            ]


            X_test=test_df[
                self.feature_columns
            ]


            self.model.train(
                X_train,
                y_train
            )


            train_proba=predict_up_probability(
                self.model,
                X_train,
            )


            best_threshold=find_best_threshold(

                y_train,

                train_proba

            )


            test_proba=predict_up_probability(
                self.model,
                X_test,
            )


            fold_signals=threshold_predictions(
                test_proba,
                best_threshold,
            )


            signals.iloc[
                train_end:test_end
            ]=fold_signals


            start+=self.step_size


        signals=signals.fillna(0)


        return signals.astype(int)
