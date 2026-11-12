"""Reusable charts and summaries for exploratory data analysis (EDA)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def plot_grade_distribution(df: pd.DataFrame, target: str = "G3", pass_mark: int = 10):
    """Histogram of final grades with the pass mark marked."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[target], bins=21, ax=ax)
    ax.axvline(pass_mark, color="red", linestyle="--", label=f"Pass mark ({pass_mark})")
    ax.set_title(f"Distribution of {target}")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_scatter_vs_grade(df: pd.DataFrame, x_col: str, target: str = "G3"):
    """Scatter plot with a trend line: how does x_col relate to the grade?"""
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.regplot(
        data=df,
        x=x_col,
        y=target,
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"},
        ax=ax,
    )
    ax.set_title(f"{x_col} vs {target}")
    fig.tight_layout()
    return fig

def plot_mean_grade_by(df: pd.DataFrame, col: str, target: str = "G3"):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x=col, y=target, errorbar=None, ax=ax)

    counts = df.groupby(col, sort=True)[target].count()
    ax.bar_label(ax.containers[0], labels=[f"n={n}" for n in counts], fontsize=8)

    ax.set(ylabel=f"Average {target}", title=f"Average {target} by {col}")
    fig.tight_layout()
    return fig

def plot_correlation_heatmap(df: pd.DataFrame):
    """Heatmap of correlations between all numeric columns."""
    corr = df.select_dtypes(include="number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))  # hide the duplicate half
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True,
                fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    return fig


def top_correlations(df: pd.DataFrame, target: str = "G3", n: int = 8) -> pd.Series:
    """The n numeric columns most correlated (up or down) with the target."""
    corr = df.select_dtypes(include="number").corr()[target].drop(target)
    ordered = corr.reindex(corr.abs().sort_values(ascending=False).index)
    return ordered.head(n).round(2)


def subject_averages(df_mat: pd.DataFrame, df_por: pd.DataFrame,
                     columns=("G1", "G2", "G3")) -> pd.DataFrame:
    """Average grades per subject (Math vs Portuguese)."""
    cols = list(columns)
    table = pd.DataFrame({
        "Math": df_mat[cols].mean(),
        "Portuguese": df_por[cols].mean(),
    }).T
    return table.round(2)


def plot_subject_averages(df_mat: pd.DataFrame, df_por: pd.DataFrame):
    """Grouped bar chart of average grades per subject."""
    table = subject_averages(df_mat, df_por)
    fig, ax = plt.subplots(figsize=(6, 4))
    table.plot(kind="bar", ax=ax)
    ax.set_ylabel("Average grade (0 to 20)")
    ax.set_ylim(0, 20)
    ax.set_title("Average grades by subject")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig