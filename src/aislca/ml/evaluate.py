"""Metrics for comparing models."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

GRADE_MIN, GRADE_MAX = 0, 20


def regression_metrics(y_true, y_pred) -> dict:
    """MAE and RMSE are in grade points (0 to 20). R2: 1 = perfect, 0 = no better than the mean."""
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(r2_score(y_true, y_pred), 3),
    }

def classification_metrics(y_true , y_pred, pos_label : int = 1) -> dict:
    """pos_label=1 scores the 'pass' class, pos_label=0 scores the 'fail' class."""

    return{
        "accuracy": round(accuracy_score(y_true,y_pred),3),
        "precision": round(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0), 3),
        "recall": round(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0), 3),
        "f1": round(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0), 3),
    }


def compare_models(fitted_models: dict, X_test, y_test, task: str, pos_label: int = 1):
    if task not in ("regression","classification"):
        raise ValueError("task must be 'regression' or 'classification'")

    rows = {}
    for name, model in fitted_models.items():
        predictions = model.predict(X_test)
        if task == "regression":
            predictions = np.clip(predictions, GRADE_MIN, GRADE_MAX)  # grades live in 0 to 20
            rows[name] = regression_metrics(y_test, predictions)
        else:
            rows[name] = classification_metrics(y_test, predictions, pos_label)


    table = pd.DataFrame(rows).T
    if task == "regression":
        return table.sort_values("rmse")

    return table.sort_values("f1",ascending=False)


def cross_validated_scores(pipelines: dict, X, y, task: str, cv_folds: int = 5,random_state: int = 42) -> pd.DataFrame:
    """Average score over several train/validation splits (more stable than one split)."""

    if task == "regression":
        cv = KFold(n_splits=cv_folds,shuffle=True,random_state=random_state)
        scoring, sign, label = "neg_root_mean_squared_error", -1, "rmse"
    else:
        cv = StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=random_state)

        scoring, sign, label = "f1",1,"f1"

    rows={}

    for name, pipeline in pipelines.items():

        scores= sign*cross_val_score(pipeline,X,y,cv=cv,scoring=scoring)
        rows[name] = {f"{label}_mean": round(scores.mean(), 3),
                      f"{label}_std": round(scores.std(), 3)}
    return pd.DataFrame(rows).T


    

