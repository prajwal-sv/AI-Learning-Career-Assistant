"""Tests for the cleaning code."""

import pandas as pd

from aislca.data import cleaning


def test_missing_report_finds_missing_values():
    students = pd.DataFrame({"age": [16, None, 17], "school": ["GP", "MS", "GP"]})

    report = cleaning.missing_report(students)

    assert list(report.index) == ["age"]
    assert report.loc["age", "missing"] == 1


def test_fix_types_makes_numbers_and_strips_text():
    students = pd.DataFrame({"age": ["16", "abc"], "school": [" GP", "MS "]})

    fixed = cleaning.fix_types(students, ["age"])

    assert fixed.loc[0, "age"] == 16
    assert pd.isna(fixed.loc[1, "age"])  # "abc" is not a number
    assert fixed.loc[0, "school"] == "GP"  # spaces removed
    assert fixed.loc[1, "school"] == "MS"


def test_fill_missing_uses_median_for_numbers():
    students = pd.DataFrame({"age": [10.0, 20.0, None, 30.0]})

    filled = cleaning.fill_missing(students)

    assert filled["age"].isna().sum() == 0
    assert filled.loc[2, "age"] == 20.0  # the middle value


def test_fill_missing_uses_most_common_for_text():
    students = pd.DataFrame({"school": ["GP", "GP", None, "MS"]})

    filled = cleaning.fill_missing(students)

    assert filled.loc[2, "school"] == "GP"


def test_remove_duplicates_removes_copies():
    students = pd.DataFrame({"age": [16, 16, 17], "school": ["GP", "GP", "MS"]})

    clean, removed = cleaning.remove_duplicates(students)

    assert removed == 1
    assert len(clean) == 2


def test_remove_invalid_rows_removes_impossible_values():
    students = pd.DataFrame({"G3": [10, 25, 0]})  # 25 is impossible
    valid_ranges = {"G3": [0, 20]}

    clean, removed = cleaning.remove_invalid_rows(students, valid_ranges)

    assert removed == 1
    assert len(clean) == 2


def test_find_outliers_counts_the_extreme_value():
    students = pd.DataFrame({"absences": [0, 1, 2, 3, 100]})

    result = cleaning.find_outliers_iqr(students, ["absences"])

    assert result.loc[0, "outliers"] == 1  # only the 100


def test_clean_student_data_gives_a_correct_report():
    config = {"cleaning": {"valid_ranges": {"age": [15, 22], "G3": [0, 20]}}}
    students = pd.DataFrame({
        "age": [16, 16, 17, 30],      # rows 0 and 1 are copies, age 30 is invalid
        "G3": [10, 10, 12, 8],
        "school": ["GP", "GP", "MS", "GP"],
    })

    clean, report = cleaning.clean_student_data(students, config)

    assert report["rows_before"] == 4
    assert report["duplicates_removed"] == 1
    assert report["invalid_rows_removed"] == 1
    assert report["rows_after"] == 2
    assert len(clean) == 2