"""Statistical tests for the student data."""

import numpy as np
import pandas as pd
from scipy import stats


def correlation_tests(df: pd.DataFrame, target: str = "G3", alpha: float = 0.05):
    """Pearson and Spearman correlation of every numeric column with the target.

    p-values below 0.0001 show as 0.0 after rounding.
    """
    rows = []
    for col in df.select_dtypes(include="number").columns:
        if col == target:
            continue
        r, p_r = stats.pearsonr(df[col], df[target])
        rho, p_s = stats.spearmanr(df[col], df[target])
        rows.append({
            "feature": col,
            "pearson_r": round(r, 3),
            "pearson_p": round(p_r, 4),
            "spearman_rho": round(rho, 3),
            "spearman_p": round(p_s, 4),
            "significant": p_r < alpha,
        })
    table = pd.DataFrame(rows)
    order = table["pearson_r"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


def compare_two_groups(df: pd.DataFrame, group_col: str, group_a, group_b,
                       target: str = "G3") -> dict:
    """Compare the target between two groups (for example higher = yes vs no)."""
    a = df.loc[df[group_col] == group_a, target]
    b = df.loc[df[group_col] == group_b, target]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least 2 students.")

    _, welch_p = stats.ttest_ind(a, b, equal_var=False)
    _, mann_p = stats.mannwhitneyu(a, b, alternative="two-sided")

    # Cohen's d: difference in means measured in standard deviations
    pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                        / (len(a) + len(b) - 2))
    cohens_d = (a.mean() - b.mean()) / pooled_sd

    return {
        "group_a": group_a, "n_a": len(a), "mean_a": round(a.mean(), 2),
        "group_b": group_b, "n_b": len(b), "mean_b": round(b.mean(), 2),
        "mean_difference": round(a.mean() - b.mean(), 2),
        "welch_t_p": round(welch_p, 4),
        "mann_whitney_p": round(mann_p, 4),
        "cohens_d": round(cohens_d, 2),
    }


def anova_by_group(df: pd.DataFrame, group_col: str, target: str = "G3") -> dict:
    """Do the average grades differ across 3+ groups (for example studytime 1 to 4)?"""
    groups = [g[target].values for _, g in df.groupby(group_col)]
    f_stat, anova_p = stats.f_oneway(*groups)
    _, kruskal_p = stats.kruskal(*groups)
    return {
        "groups": len(groups),
        "f_statistic": round(f_stat, 2),
        "anova_p": round(anova_p, 4),
        "kruskal_p": round(kruskal_p, 4),
    }


def chi_square_test(df: pd.DataFrame, col_a: str, col_b: str):
    """Are two categorical columns related? Returns (result_dict, count_table)."""
    table = pd.crosstab(df[col_a], df[col_b])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    result = {
        "chi2": round(chi2, 2),
        "p_value": round(p, 4),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(cramers_v, 2),
    }
    return result, table