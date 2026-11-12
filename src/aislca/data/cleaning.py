"""Clean the raw student data."""

import pandas as pd

from aislca.data.ingestion import PROJECT_ROOT ,load_config,load_student_data

# Columns where we look for outliers (the others are small 1-5 scales)
OUTLIER_COLUMNS = ["age", "absences", "G1", "G2", "G3"]

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Show only the columns that have missing values."""

    counts = df.isna().sum()
    report = pd.DataFrame(
        {"missing" : counts , "percent":(counts/len(df)*100).round(2)}

    )
    return report[report["missing"] > 0 ]


def fix_types(df: pd.DataFrame , numeric_columns : list) -> pd.DataFrame:
    """Strip spaces from text columns and make numeric columns numeric."""
    df = df.copy()
    for col in df.columns : 
        if col in numeric_columns:
            df[col] = pd.to_numeric(df[col],errors="coerce")
        else:
            df[col] = df[col].astype(str).str.strip()

    return df


def fill_missing( df : pd.DataFrame) -> pd.DataFrame :
    """Numbers get the median, text gets the most common value."""

    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    return df

def remove_duplicates(df: pd.DataFrame):
    """Drop rows that are exact copies. Returns (clean_df, number_removed)."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    return df, before - len(df)

def remove_invalid_rows(df: pd.DataFrame, valid_ranges: dict):
    """Drop rows with values outside the allowed ranges."""
    start_rows = len(df)
    
    for col, (low, high) in valid_ranges.items():
        if col in df.columns:
            df = df[df[col].between(low, high)]
            
    dropped = start_rows - len(df)
    return df.reset_index(drop=True), dropped


def find_outliers_iqr(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Count values beyond 1.5 x IQR from the quartiles. Does NOT delete."""
    rows = []

    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append(
            {"column": col, "lower_limit": low, "upper_limit": high, "outliers": count}
        )
    return pd.DataFrame(rows)


def clean_student_data(df: pd.DataFrame, config: dict = None):
    """Run every cleaning step. Returns (clean_df, report_dict)."""

    config = config or load_config()
    valid_ranges = config["cleaning"]["valid_ranges"]
    numeric_columns = list(valid_ranges.keys())

    report = {"rows_before": len(df)}

    df = fix_types(df, numeric_columns)
    report["missing_values_filled"] = int(df.isna().sum().sum())
    df = fill_missing(df)
    df, report["duplicates_removed"] = remove_duplicates(df)
    df, report["invalid_rows_removed"] = remove_invalid_rows(df, valid_ranges)

    report["rows_after"] = len(df)
    return df, report

    
def run_cleaning() -> None:
    """Clean both raw files and save them to data/processed/."""
    config = load_config()
    out_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for subject in ("mat", "por"):
        raw = load_student_data(subject, config)
        clean, report = clean_student_data(raw, config)
        out_path = out_dir / f"student_{subject}_clean.csv"
        clean.to_csv(out_path, index=False)
        print(f"[{subject}] {report}")
        print(f"[{subject}] saved to {out_path}")


if __name__ == "__main__":
    run_cleaning()

