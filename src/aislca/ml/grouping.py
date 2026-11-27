"""Student groups (V2).

K-Means puts students into groups with similar study habits and grades.
We choose the number of groups (k) with the silhouette score, describe every
group in neutral words, and can place a new student in a group.

Groups are descriptions of patterns in this dataset. They are not labels of
people, not a ranking, and not a prediction.
"""

import json

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from aislca.data.ingestion import PROJECT_ROOT, load_config, load_processed_data

LETTERS = "ABCDEFGHIJKL"

# Plain words for each input column
FEATURE_WORDS = {
    "studytime": "weekly study time",
    "absences": "absences",
    "failures": "earlier class failures",
    "G1": "first-period grades",
    "G2": "second-period grades",
}


def get_group_settings(config=None):
    """Read the group settings from the config file."""
    config = config or load_config()
    settings = dict(config["groups"])
    settings["random_state"] = config["ml"]["random_state"]
    return settings


def prepare_group_inputs(students, features, absences_cap):
    """Pick the feature columns and cap very high absences."""
    inputs = students[features].copy()
    if "absences" in inputs.columns and absences_cap is not None:
        inputs["absences"] = inputs["absences"].clip(upper=absences_cap)
    return inputs.astype(float)


def count_group_sizes(labels):
    """Count the students in each group. Returns {group_number: students}."""
    sizes = {}
    for label in labels:
        number = int(label)
        sizes[number] = sizes.get(number, 0) + 1
    return sizes


def score_cluster_counts(scaled, k_values, random_state):
    """Try each k and record its silhouette score and smallest group."""
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(scaled)
        sizes = count_group_sizes(labels)
        rows.append({
            "k": k,
            "silhouette": round(float(silhouette_score(scaled, labels)), 3),
            "smallest_group": min(sizes.values()),
            "inertia": round(float(model.inertia_), 1),
        })
    return rows


def choose_k(rows, total_students, min_group_share):
    """The k with the best silhouette among those without tiny groups.

    On a tie the smaller k wins (fewer groups are easier to understand).
    """
    smallest_allowed = min_group_share * total_students

    best_row = None
    for row in rows:
        if row["smallest_group"] < smallest_allowed:
            continue
        if best_row is None or row["silhouette"] > best_row["silhouette"]:
            best_row = row

    # If no k follows the rule, fall back to the best silhouette overall
    if best_row is None:
        for row in rows:
            if best_row is None or row["silhouette"] > best_row["silhouette"]:
                best_row = row

    return best_row["k"]


def name_groups(labels):
    """Neutral names: Group A is the biggest group, then B, C, ...

    The order is by size only. It says nothing about quality.
    """
    sizes = count_group_sizes(labels)
    ordered_numbers = sorted(sizes, key=sizes.get, reverse=True)   # biggest first

    names = {}
    position = 0
    for number in ordered_numbers:
        names[number] = "Group " + LETTERS[position]
        position += 1
    return names


def describe_level(group_mean, overall_mean, overall_std, gap):
    """Say whether a group's average is higher, lower or about the overall average."""
    if overall_std == 0:
        return "about average"
    difference = (group_mean - overall_mean) / overall_std
    if difference >= gap:
        return "higher than average"
    if difference <= -gap:
        return "lower than average"
    return "about average"


def build_group_profiles(students, labels, names, features, gap):
    """Describe each group in plain, neutral words (original units, not scaled)."""
    overall_mean = students[features].mean()
    overall_std = students[features].std()
    total = len(students)

    profiles = []
    for number, group_name in names.items():
        members = students[labels == number]   # the students in this group

        parts = []
        details = {}
        for feature in features:
            group_mean = float(members[feature].mean())
            level = describe_level(
                group_mean, float(overall_mean[feature]), float(overall_std[feature]), gap)
            words = FEATURE_WORDS.get(feature, feature)
            parts.append(words + " " + level)
            details[feature] = {
                "group_average": round(group_mean, 2),
                "all_students_average": round(float(overall_mean[feature]), 2),
                "level": level,
            }

        profiles.append({
            "name": group_name,
            "group_number": number,
            "students": len(members),
            "share_percent": round(100 * len(members) / total, 1),
            "description": "Compared with all students: " + "; ".join(parts) + ".",
            "details": details,
        })
    return profiles


def run_grouping():
    """Choose k, train the groups, and save the model and a small report."""
    config = load_config()
    settings = get_group_settings(config)
    features = settings["features"]

    # Step 1: load the cleaned Math students and prepare the inputs
    students = load_processed_data("mat", config)
    inputs = prepare_group_inputs(students, features, settings["absences_cap"])

    # Step 2: put all features on the same scale (K-Means compares distances)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(inputs)

    # Step 3: try several k and choose one with the silhouette score
    k_values = range(settings["k_min"], settings["k_max"] + 1)
    rows = score_cluster_counts(scaled, k_values, settings["random_state"])
    best_k = choose_k(rows, len(students), settings["min_group_share"])

    # Step 4: train the final groups
    model = KMeans(n_clusters=best_k, n_init=10, random_state=settings["random_state"])
    labels = model.fit_predict(scaled)

    # Step 5: name and describe the groups
    names = name_groups(labels)
    profiles = build_group_profiles(
        students, labels, names, features, settings["description_gap"])

    # Typical values, used when a new student leaves a question empty
    defaults = {}
    for feature in features:
        defaults[feature] = float(students[feature].median())

    # Step 6: save the model and a small report
    bundle = {
        "features": features,
        "absences_cap": settings["absences_cap"],
        "scaler": scaler,
        "model": model,
        "names": names,
        "profiles": profiles,
        "defaults": defaults,
        "chosen_k": best_k,
    }
    model_path = PROJECT_ROOT / settings["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)

    report = {
        "dataset": "UCI Student Performance (Math)",
        "students": len(students),
        "features": features,
        "chosen_k": best_k,
        "silhouette_by_k": rows,
        "groups": profiles,
        "note": ("Groups describe patterns in this dataset. They are not labels of "
                 "people, not a ranking, and not a prediction."),
    }
    report_path = PROJECT_ROOT / settings["report_path"]
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # Print a short summary
    print("k  silhouette  smallest group  inertia")
    for row in rows:
        print(f"{row['k']}  {row['silhouette']:>10}  {row['smallest_group']:>14}  {row['inertia']:>7}")
    print()
    print("Chosen k:", best_k)
    print()
    for profile in profiles:
        print(f"{profile['name']}: {profile['students']} students ({profile['share_percent']}%)")
        print("  " + profile["description"])
    print()
    print("Model saved to: ", model_path)
    print("Report saved to:", report_path)
    return report


def load_group_bundle(config=None):
    """Load the saved group model."""
    config = config or load_config()
    path = PROJECT_ROOT / config["groups"]["model_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"Group model not found: {path}. Run: python -m aislca.ml.grouping"
        )
    return joblib.load(path)


def assign_group(bundle, student):
    """Put one new student in a group.

    student is a dictionary such as {"studytime": 1, "absences": 12, "failures": 1,
    "G1": 8, "G2": 7}. Missing answers use typical values, and we say which.
    """
    row = {}
    used_defaults = []
    for feature in bundle["features"]:
        value = student.get(feature)
        if value is None:
            row[feature] = bundle["defaults"][feature]
            used_defaults.append(feature)
        else:
            row[feature] = float(value)

    inputs = prepare_group_inputs(
        pd.DataFrame([row]), bundle["features"], bundle["absences_cap"])
    scaled = bundle["scaler"].transform(inputs)
    number = int(bundle["model"].predict(scaled)[0])
    group_name = bundle["names"][number]

    description = ""
    for profile in bundle["profiles"]:
        if profile["name"] == group_name:
            description = profile["description"]

    return {
        "group": group_name,
        "description": description,
        "used_defaults": used_defaults,
    }


if __name__ == "__main__":
    run_grouping()