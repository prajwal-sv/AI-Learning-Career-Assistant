"""Group students with K-Means clustering."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def scale_columns(students, columns):
    """Pick the columns and put them on the same scale.

    K-Means measures distances, so big numbers (absences) must not
    overpower small numbers (studytime).
    """
    chosen_columns = students[columns]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(chosen_columns)
    return scaled_data, scaler


def try_different_k(scaled_data, k_values=range(2, 9), random_state=42):
    """Run K-Means for several values of k and score each one.

    inertia: how tight the groups are (lower is tighter, but it always
             goes down when k goes up).
    silhouette: how well separated the groups are (higher is better).
    """
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(scaled_data)
        rows.append({
            "k": k,
            "inertia": round(model.inertia_, 1),
            "silhouette": round(silhouette_score(scaled_data, labels), 3),
        })
    return pd.DataFrame(rows)


def run_kmeans(scaled_data, n_clusters=4, random_state=42):
    """Group the students. Returns the model and one group number per student."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(scaled_data)
    return model, labels


def describe_clusters(students, labels, columns, pass_mark=10):
    """Make a table with the average of each column for every group."""
    students = students.copy()
    students["cluster"] = labels
    students["passed"] = (students["G3"] >= pass_mark).astype(int)

    summary = students.groupby("cluster")[columns].mean().round(2)
    summary["students"] = students.groupby("cluster").size()
    summary["average_G3"] = students.groupby("cluster")["G3"].mean().round(2)
    summary["pass_rate"] = students.groupby("cluster")["passed"].mean().round(2)
    return summary


def plot_k_choice(results):
    """Draw the elbow chart and the silhouette chart side by side."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))

    left.plot(results["k"], results["inertia"], marker="o")
    left.set_title("Elbow: inertia (lower = tighter groups)")
    left.set_xlabel("Number of groups (k)")

    right.plot(results["k"], results["silhouette"], marker="o")
    right.set_title("Silhouette (higher = better separated)")
    right.set_xlabel("Number of groups (k)")

    fig.tight_layout()
    return fig


def plot_clusters(scaled_data, labels):
    """Draw the students as dots, colored by group.

    We have 5 columns but a screen has 2 directions, so PCA squashes
    the 5 columns into 2 while keeping as much of the pattern as possible.
    """
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(scaled_data)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", alpha=0.7)
    ax.legend(*scatter.legend_elements(), title="Group")
    ax.set_title("Student groups (squashed to 2 directions)")
    ax.set_xlabel("Direction 1")
    ax.set_ylabel("Direction 2")
    fig.tight_layout()
    return fig