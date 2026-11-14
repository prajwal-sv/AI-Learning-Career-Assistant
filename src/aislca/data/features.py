"""Prepare student data for the models."""

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "G3"
PRIOR_GRADES = ["G1", "G2"]


def add_pass_label(students, pass_mark=10):
    """Add a column 'passed': 1 if G3 is at least the pass mark, otherwise 0."""
    students = students.copy()
    students["passed"] = (students[TARGET] >= pass_mark).astype(int)
    return students


def get_feature_columns(students, include_prior_grades=True):
    """Pick the input columns (never the target and never the pass label)."""
    feature_columns = []
    for column in students.columns:
        if column == TARGET or column == "passed":
            continue
        if column in PRIOR_GRADES and not include_prior_grades:
            continue
        feature_columns.append(column)
    return feature_columns


def split_data(students, pass_mark=10, test_size=0.2, random_state=42):
    """Split students into train data and test data."""
    students = add_pass_label(students, pass_mark)
    train_data, test_data = train_test_split(
        students,
        test_size=test_size,
        random_state=random_state,
        stratify=students["passed"],  # keeps the pass rate similar in both parts
    )
    return train_data, test_data


def prepare_inputs(students, feature_columns, training_columns=None):
    """Turn the chosen columns into numbers only.

    Text columns such as yes/no become 0/1 columns.
    training_columns: the column list from the training data. Pass it for new data
    so the columns match the training data exactly.
    """
    inputs = pd.get_dummies(students[feature_columns]).astype(float)

    if training_columns is not None:
        inputs = inputs.reindex(columns=training_columns, fill_value=0.0)

    return inputs



















    