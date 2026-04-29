import pandas as pd


def normalize_signal_policy(policy=None, fallback_threshold: float = 0.55) -> dict:
    if isinstance(policy, dict):
        entry_threshold = float(
            policy.get(
                "entry_threshold",
                policy.get("threshold", fallback_threshold),
            )
        )
        exit_threshold = float(
            policy.get(
                "exit_threshold",
                entry_threshold,
            )
        )
    elif policy is None:
        entry_threshold = float(fallback_threshold)
        exit_threshold = float(fallback_threshold)
    else:
        entry_threshold = float(policy)
        exit_threshold = float(policy)

    if exit_threshold > entry_threshold:
        exit_threshold = entry_threshold

    return {
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
    }


def generate_probability_signals(probabilities, policy=None) -> pd.Series:
    policy = normalize_signal_policy(policy)

    probabilities = pd.Series(probabilities).astype(float)

    entry_threshold = float(policy["entry_threshold"])
    exit_threshold = float(policy["exit_threshold"])

    if entry_threshold <= exit_threshold:
        return (probabilities > entry_threshold).astype(int)

    signals = []
    in_position = 0

    for probability in probabilities:
        if pd.isna(probability):
            signals.append(in_position)
            continue

        if in_position == 1 and probability < exit_threshold:
            in_position = 0
        elif in_position == 0 and probability > entry_threshold:
            in_position = 1

        signals.append(in_position)

    return pd.Series(signals, index=probabilities.index, dtype=int)
