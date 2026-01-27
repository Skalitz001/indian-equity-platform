import pandas as pd
from src.strategies.sma_crossover import SMACrossoverStrategy


def test_sma_strategy_output_shape():
    df = pd.DataFrame({
        "sma_20": [1, 2, 3],
        "sma_50": [3, 2, 1],
    })

    strategy = SMACrossoverStrategy(20, 50)
    signals = strategy.generate_signals(df)

    assert len(signals) == len(df)
    assert set(signals.unique()).issubset({0, 1})
