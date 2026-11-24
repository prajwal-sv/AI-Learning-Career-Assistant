"""Load the saved model and make predictions for new students."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aislca.data.features import prepare_inputs
from aislca.data.ingestion import PROJECT_ROOT, load_config


def load_model_bundle(model_path=None):
    """Load the saved file (models, column lists and default values)."""
    if model_path is None:
        config = load_config()
        model_path = PROJECT_ROOT / config["registry"]["model_path"]

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Create it with: python -m aislca.ml.train"
        )
    return joblib.load(model_path)


def predict_students(bundle, students):
    """Predict for a table of students (one row per student).

    Any input column missing from the table gets its typical (default) value.
    """
    students = students.copy()

    # Fill the questions that were not answered
    for column in bundle["feature_columns"]:
        if column not in students.columns:
            students[column] = bundle["defaults"][column]

    # Turn the answers into numbers, in the same column order as the training data
    inputs = prepare_inputs(students, bundle["feature_columns"], bundle["training_columns"])

    # Predict the mark (between 0 and 20) and the chance of passing
    marks = bundle["regression_model"].predict(inputs)
    marks = np.clip(marks, 0, 20).round(1)
    pass_chance = bundle["classification_model"].predict_proba(inputs)[:, 1]

    results = pd.DataFrame({
        "estimated_mark": marks,
        "pass_chance": pass_chance.round(2),
    })
    results["estimate"] = np.where(
        pass_chance >= 0.5, "Estimated to pass", "Estimated below the pass mark"
    )
    return results


def predict_one(bundle, student):
    """Predict for one student given as a dictionary, like {'G1': 8, 'G2': 9}."""
    table = pd.DataFrame([student])
    results = predict_students(bundle, table)
    return results.iloc[0].to_dict()