from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


ADABOOST_GRID = (
    (100, 0.05, 1, 10),
    (200, 0.05, 1, 10),
    (100, 0.10, 2, 20),
    (200, 0.10, 2, 20),
)


def build_adaboost_candidates():
    candidates = []

    for n_estimators, learning_rate, max_depth, min_leaf in ADABOOST_GRID:
        config = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_samples_leaf": min_leaf,
        }

        label = (
            f"ada_ne{n_estimators}_"
            f"lr{learning_rate}_"
            f"md{max_depth}_"
            f"ml{min_leaf}"
        )

        def model_builder(config=config):
            return AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=config["max_depth"],
                    min_samples_leaf=config["min_samples_leaf"],
                    random_state=42,
                ),
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                random_state=42,
            )

        candidates.append({
            "label": label,
            "config": config,
            "builder": model_builder,
        })

    return candidates
