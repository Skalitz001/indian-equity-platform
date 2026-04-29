import pandas as pd

from src.strategies.signal_policy import (
    generate_probability_signals,
    normalize_signal_policy,
)


def test_normalize_signal_policy_clamps_exit_threshold():
    policy = normalize_signal_policy(
        {
            "entry_threshold": 0.62,
            "exit_threshold": 0.70,
        }
    )

    assert policy == {
        "entry_threshold": 0.62,
        "exit_threshold": 0.62,
    }


def test_generate_probability_signals_supports_hysteresis():
    probabilities = pd.Series([0.63, 0.59, 0.57, 0.61, 0.52, 0.58])

    signals = generate_probability_signals(
        probabilities,
        {
            "entry_threshold": 0.60,
            "exit_threshold": 0.55,
        },
    )

    assert signals.tolist() == [1, 1, 1, 1, 0, 0]
