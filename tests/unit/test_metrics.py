"""Tests for the scoring code."""

import json

import pandas as pd
import pytest

from aislca.ml import evaluate


def test_regression_metrics_for_perfect_predictions():
    result = evaluate.regression_metrics([10, 12, 14], [10, 12, 14])

    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0


def test_regression_metrics_for_a_known_error():
    # Every prediction is wrong by exactly 2 points
    result = evaluate.regression_metrics([10, 10], [12, 8])

    assert result["mae"] == 2.0
    assert result["rmse"] == 2.0


def test_classification_metrics_for_the_pass_group():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]  # one passing student was missed

    result = evaluate.classification_metrics(y_true, y_pred, pos_label=1)

    assert result["accuracy"] == 0.75
    assert result["precision"] == 1.0  # every predicted pass was right
    assert result["recall"] == 0.5     # found 1 of 2 passing students
    assert result["f1"] == 0.667


def test_classification_metrics_for_the_fail_group():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]

    result = evaluate.classification_metrics(y_true, y_pred, pos_label=0)

    assert result["recall"] == 1.0  # found both failing students


def test_confusion_matrix_counts_are_correct():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 0]

    table = evaluate.confusion_matrix_table(y_true, y_pred)

    assert table.loc["Actually failed", "Predicted fail"] == 2
    assert table.loc["Actually passed", "Predicted fail"] == 1
    assert table.loc["Actually passed", "Predicted pass"] == 1


def test_pick_best_model_for_regression_chooses_the_lowest_error():
    cv_table = pd.DataFrame(
        {"average": [4.5, 2.0, 3.0], "spread": [0.1, 0.1, 0.1]},
        index=["Baseline (mean)", "Linear Regression", "Random Forest"],
    )

    assert evaluate.pick_best_model(cv_table, "regression") == "Linear Regression"


def test_pick_best_model_for_classification_ignores_the_baseline():
    cv_table = pd.DataFrame(
        {"average": [0.9, 0.7, 0.85], "spread": [0.1, 0.1, 0.1]},
        index=["Baseline (always pass)", "Logistic Regression", "Random Forest"],
    )

    # The baseline has the top score, but it must never win
    assert evaluate.pick_best_model(cv_table, "classification") == "Random Forest"


def test_compare_models_rejects_an_unknown_task():
    with pytest.raises(ValueError):
        evaluate.compare_models({}, None, None, "banana")


def test_save_metrics_json_writes_a_readable_file(tmp_path):
    path = tmp_path / "new_folder" / "metrics.json"

    evaluate.save_metrics_json({"accuracy": 0.9}, path)

    with open(path, encoding="utf-8") as file:
        saved = json.load(file)
    assert saved == {"accuracy": 0.9}