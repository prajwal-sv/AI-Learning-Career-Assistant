"""Hyperparameter tuning helpers (V2).

A hyperparameter is a setting chosen BEFORE training, for example how deep
a tree may grow. We try many combinations with cross-validation on the
training students only.
"""

from sklearn.model_selection import RandomizedSearchCV


def get_search_grids(task):
    """The settings to try for each model.

    The lists are small on purpose: we only have 316 training students,
    so a huge search would just find luck.
    """
    decision_tree = {
        "max_depth": [2, 3, 4, 5, 6, 8],
        "min_samples_leaf": [1, 2, 5, 10],
    }
    random_forest = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 4, 6, 8],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [1.0, 0.5, "sqrt"],
    }
    xgboost = {
        "n_estimators": [100, 200, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    grids = {
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
        "XGBoost": xgboost,
    }

    # Only the pass/fail task has Logistic Regression.
    # C controls how strongly the model is kept simple (smaller = simpler).
    if task == "classification":
        grids["Logistic Regression"] = {"C": [0.01, 0.1, 1, 10, 100]}

    return grids


def count_combinations(grid):
    """How many different combinations a grid contains."""
    total = 1
    for values in grid.values():
        total = total * len(values)
    return total


def build_searches(models, task, n_iter, inner_folds, random_state):
    """Wrap each model that has a grid in a RandomizedSearchCV.

    Returns a dictionary like {"Random Forest (tuned)": search, ...}.
    Each search can be used like a normal model: fit() finds the best
    settings and then trains the best version.
    """
    if task == "regression":
        scoring = "neg_root_mean_squared_error"
    else:
        scoring = "f1"

    grids = get_search_grids(task)
    searches = {}
    for name, grid in grids.items():
        # Never ask for more tries than there are combinations
        tries = min(n_iter, count_combinations(grid))
        search = RandomizedSearchCV(
            estimator=models[name],
            param_distributions=grid,
            n_iter=tries,
            scoring=scoring,
            cv=inner_folds,
            random_state=random_state,
            n_jobs=-1,
        )
        searches[name + " (tuned)"] = search
    return searches


def clean_params(params):
    """Turn numpy numbers into plain Python numbers so they save as JSON."""
    cleaned = {}
    for key, value in params.items():
        if hasattr(value, "item"):
            value = value.item()
        cleaned[key] = value
    return cleaned


def split_fitted_model(model):
    """Return (model, settings).

    If the model is a finished search, we keep only its best model and the
    best settings it found. Otherwise we return the model and no settings.
    """
    if hasattr(model, "best_estimator_"):
        return model.best_estimator_, clean_params(model.best_params_)
    return model, {}