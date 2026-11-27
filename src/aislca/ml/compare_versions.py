"""Write a plain V1 vs V2 comparison to docs/v1_vs_v2.md.

Run it with:  python -m aislca.ml.compare_versions
It only reads saved files. It trains nothing.
"""

import json

from aislca.data.ingestion import PROJECT_ROOT

V1_PATH = PROJECT_ROOT / "models" / "registry" / "v1_metrics.json"
V2_PATH = PROJECT_ROOT / "models" / "registry" / "metrics.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "v1_vs_v2.md"


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def make_table(headers, rows):
    """Build a small markdown table from a list of rows."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for row in rows:
        cells = []
        for cell in row:
            cells.append(str(cell))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def change(old, new):
    """New minus old, as text with a sign."""
    return f"{new - old:+.3f}"


def counts_from_confusion(confusion):
    """Pull caught / missed / false alarms out of a confusion matrix."""
    counts = confusion["counts"]
    return {
        "caught": counts[0][0],
        "missed": counts[0][1],
        "false_alarms": counts[1][0],
    }


def build_report(v1, v2):
    v1_reg = v1["regression"]
    v2_reg = v2["regression"]
    v1_cls = v1["classification"]
    v2_cls = v2["classification"]
    support = v2_cls["support_analysis"]

    lines = []
    lines.append("# V1 vs V2")
    lines.append("")
    lines.append("Both versions use the same 316 training and 79 test students.")
    lines.append("")

    # Are the final models and scores the same?
    same_models = (v1_reg["best_model"] == v2_reg["best_model"]
                   and v1_cls["best_model"] == v2_cls["best_model"])
    same_scores = (v1_reg["test_scores"] == v2_reg["test_scores"]
                   and v1_cls["test_scores_fail_group"] == v2_cls["test_scores_fail_group"])
    if same_models and same_scores:
        lines.append("**The final models and their test scores are identical to V1.** "
                     "XGBoost and tuning did not beat the V1 models in cross-validation, "
                     "so the V1 models were kept.")
    else:
        lines.append("The final models or test scores differ from V1. See the tables below.")
    lines.append("")

    # Table 1: the mark model
    lines.append("## Predicting the final mark (test students)")
    lines.append("")
    rows = [["Model", v1_reg["best_model"], v2_reg["best_model"], ""]]
    for key in ["mae", "rmse", "r2"]:
        old = v1_reg["test_scores"][key]
        new = v2_reg["test_scores"][key]
        rows.append([key.upper(), old, new, change(old, new)])
    lines.append(make_table(["Measure", "V1", "V2", "Change"], rows))
    lines.append("")

    # Table 2: finding students who may need support
    lines.append("## Finding students who may need support (test students)")
    lines.append("")
    v1_counts = counts_from_confusion(v1_cls["confusion_matrix"])
    v2_counts = counts_from_confusion(v2_cls["confusion_matrix"])
    chosen = support["test_at_chosen_threshold"]
    v1_fail = v1_cls["test_scores_fail_group"]
    v2_fail = v2_cls["test_scores_fail_group"]

    heading = "V2 (threshold " + str(support["chosen_threshold"]) + ")"
    rows = [
        ["Model", v1_cls["best_model"], v2_cls["best_model"], v2_cls["best_model"]],
        ["Failing students found (recall)", v1_fail["recall"], v2_fail["recall"], chosen["recall"]],
        ["Flags that were right (precision)", v1_fail["precision"], v2_fail["precision"], chosen["precision"]],
        ["F1 (fail group)", v1_fail["f1"], v2_fail["f1"], chosen["f1"]],
        ["Failing students caught", v1_counts["caught"], v2_counts["caught"], chosen["caught"]],
        ["Failing students missed", v1_counts["missed"], v2_counts["missed"], chosen["missed"]],
        ["False alarms", v1_counts["false_alarms"], v2_counts["false_alarms"], chosen["false_alarms"]],
    ]
    lines.append(make_table(["Measure", "V1 (threshold 0.5)", "V2 (threshold 0.5)", heading], rows))
    lines.append("")
    lines.append("The V2 threshold was chosen using the training students only: "
                 "the highest threshold that still found at least "
                 + str(support["target_recall"]) + " of the failing students.")
    lines.append("")

    # How big is one student?
    failed_total = v2_counts["caught"] + v2_counts["missed"]
    test_total = v2_cls["confusion_matrix"]["counts"][0][0] + v2_cls["confusion_matrix"]["counts"][0][1] \
        + v2_cls["confusion_matrix"]["counts"][1][0] + v2_cls["confusion_matrix"]["counts"][1][1]
    lines.append("## How much can we trust small differences?")
    lines.append("")
    lines.append("The test set has " + str(test_total) + " students, " + str(failed_total)
                 + " of whom failed. One failing student moves recall by about "
                 + str(round(100 / failed_total, 1)) + " percentage points, and one student "
                 "moves accuracy by about " + str(round(100 / test_total, 1))
                 + ". Differences of one or two students are noise, not proof of a better model.")
    lines.append("")

    # Table 3: model comparison on the fail group (training students)
    lines.append("## Model comparison on the fail group (training students, cross-validated)")
    lines.append("")
    rows = []
    for name, scores in support["model_comparison_training"].items():
        rows.append([name, scores["recall"], scores["precision"], scores["f1"], scores["roc_auc"]])
    lines.append(make_table(["Model", "Recall", "Precision", "F1", "ROC AUC"], rows))
    lines.append("")

    # Table 4: the threshold table
    lines.append("## Threshold analysis (training students, cross-validated)")
    lines.append("")
    lines.append("A student is flagged when the estimated fail probability is at or above the threshold.")
    lines.append("")
    rows = []
    for row in support["threshold_table_training"]:
        rows.append([row["threshold"], row["flagged"], row["caught"], row["missed"],
                     row["false_alarms"], row["recall"], row["precision"], row["f1"]])
    lines.append(make_table(
        ["Threshold", "Flagged", "Caught", "Missed", "False alarms", "Recall", "Precision", "F1"], rows))
    lines.append("")
    lines.append("Predictions are estimates, not guarantees. They should support teachers "
                 "and counselors, not replace them.")
    lines.append("")
    return "\n".join(lines)


def main():
    v1 = read_json(V1_PATH)
    v2 = read_json(V2_PATH)
    report = build_report(v1, v2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    print(report)
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()