"""Tests for the feature code."""

import pandas as pd

from aislca.data import features


def test_add_pass_label_uses_the_pass_mark():
    students = pd.DataFrame({"G3": [9, 10, 15]})

    result = features.add_pass_label(students, pass_mark=10)

    assert list(result["passed"]) == [0, 1, 1]
    assert "passed" not in students.columns  # the original table is not changed


def test_get_feature_columns_leaves_out_the_target():
    students = pd.DataFrame(columns=["age", "G1", "G2", "G3", "passed"])

    columns = features.get_feature_columns(students)

    assert columns == ["age", "G1", "G2"]


def test_get_feature_columns_can_leave_out_prior_grades():
    students = pd.DataFrame(columns=["age", "G1", "G2", "G3", "passed"])

    columns = features.get_feature_columns(students, include_prior_grades=False)

    assert columns == ["age"]


def test_split_data_gives_the_right_sizes_and_no_overlap():
    students = pd.DataFrame({"age": range(20), "G3": [5] * 8 + [15] * 12})

    train_data, test_data = features.split_data(
        students, pass_mark=10, test_size=0.25, random_state=42)

    assert len(train_data) == 15
    assert len(test_data) == 5
    assert "passed" in train_data.columns
    assert len(set(train_data.index) & set(test_data.index)) == 0  # no student in both


def test_split_data_is_repeatable():
    students = pd.DataFrame({"age": range(20), "G3": [5] * 8 + [15] * 12})

    train_one, _ = features.split_data(students, 10, 0.25, 42)
    train_two, _ = features.split_data(students, 10, 0.25, 42)

    assert train_one.index.equals(train_two.index)


def test_prepare_inputs_turns_text_into_numbers():
    students = pd.DataFrame({"age": [16, 17], "school": ["GP", "MS"]})

    inputs = features.prepare_inputs(students, ["age", "school"])

    assert list(inputs.columns) == ["age", "school_GP", "school_MS"]
    assert inputs.loc[0, "school_GP"] == 1.0
    assert inputs.loc[0, "school_MS"] == 0.0


def test_prepare_inputs_matches_the_training_columns():
    new_student = pd.DataFrame({"age": [16], "school": ["GP"]})
    training_columns = ["age", "school_GP", "school_MS"]

    inputs = features.prepare_inputs(new_student, ["age", "school"], training_columns)

    assert list(inputs.columns) == training_columns
    assert inputs.loc[0, "school_MS"] == 0.0  # missing column is added as 0