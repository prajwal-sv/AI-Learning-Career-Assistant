"""Join the Math and Portuguese students who appear in both files (V2).

The UCI download has no student ID. The authors' R script
(data/raw/student-merge.R) matches students on 13 answers. We do the same.
We start from the CLEANED files in data/processed/. Raw files are never edited.
"""

import pandas as pd

from aislca.data.ingestion import PROJECT_ROOT, load_config, load_processed_data


def drop_repeated_keys(students, keys):
    """Remove students who share all their key answers with another student.

    If two students in the same file give identical answers, we cannot tell
    which record in the other file belongs to which of them, so we leave both out.
    Returns (kept_students, number_removed).
    """
    repeated = students.duplicated(subset=keys, keep=False)   # True for every repeated row
    kept = students[~repeated].reset_index(drop=True)         # ~ means "not"
    return kept, int(repeated.sum())


def merge_subjects(math_students, portuguese_students, keys):
    """Join the two files on the key answers.

    Columns that exist in both files get the endings _mat and _por,
    for example G2_mat and G2_por. Returns (merged_students, report).
    """
    math_kept, math_removed = drop_repeated_keys(math_students, keys)
    por_kept, por_removed = drop_repeated_keys(portuguese_students, keys)

    merged = pd.merge(math_kept, por_kept, on=keys, suffixes=("_mat", "_por"))

    report = {
        "math_students": len(math_students),
        "portuguese_students": len(portuguese_students),
        "math_removed_repeated_keys": math_removed,
        "portuguese_removed_repeated_keys": por_removed,
        "merged_students": len(merged),
    }
    return merged, report


def load_merged_data(config=None):
    """Load the merged dataset from data/processed/."""
    config = config or load_config()
    path = PROJECT_ROOT / config["data"]["processed_dir"] / config["merge"]["output_file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Merged data not found: {path}. Run: python -m aislca.data.merge"
        )
    return pd.read_csv(path)


def run_merge():
    """Merge the cleaned files and save the result."""
    config = load_config()
    keys = config["merge"]["keys"]

    math_students = load_processed_data("mat", config)
    portuguese_students = load_processed_data("por", config)
    merged, report = merge_subjects(math_students, portuguese_students, keys)

    out_path = PROJECT_ROOT / config["data"]["processed_dir"] / config["merge"]["output_file"]
    merged.to_csv(out_path, index=False)

    print(report)
    print("Saved to:", out_path)


if __name__ == "__main__":
    run_merge()