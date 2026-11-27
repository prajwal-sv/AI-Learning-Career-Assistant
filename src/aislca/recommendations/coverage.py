"""Check how often each rule fires on the merged students.

A rule that fires for almost everyone (or for nobody) is not useful, so this
helps us judge the limits in configs/config.yaml.
Run it with:  python -m aislca.recommendations.coverage
"""

from aislca.data.merge import load_merged_data
from aislca.recommendations.rules import load_rule_settings, recommend


def student_from_merged_row(row):
    """Build the rule input for one merged student.

    Current marks are the latest period grades (G2) in each subject.
    The study-time answer comes from the Math record.
    """
    return {
        "marks": {
            "Mathematics": float(row["G2_mat"]),
            "Portuguese": float(row["G2_por"]),
        },
        "study_time": int(row["studytime_mat"]),
        "absences": {
            "Mathematics": float(row["absences_mat"]),
            "Portuguese": float(row["absences_por"]),
        },
        "past_failures": {
            "Mathematics": float(row["failures_mat"]),
            "Portuguese": float(row["failures_por"]),
        },
    }


def rule_coverage(merged, settings):
    """Count how many students get each rule. Returns {label: students}."""
    counts = {}
    for _, row in merged.iterrows():
        student = student_from_merged_row(row)

        # Each student counts once per rule, even with two subjects
        labels_for_student = []
        for item in recommend(student, settings):
            label = item["rule"] + " (" + item["area"] + ")"
            if label not in labels_for_student:
                labels_for_student.append(label)

        for label in labels_for_student:
            counts[label] = counts.get(label, 0) + 1
    return counts


def main():
    merged = load_merged_data()
    settings = load_rule_settings()
    counts = rule_coverage(merged, settings)

    total = len(merged)
    print("Merged students:", total)
    print()
    for label in sorted(counts):
        share = 100 * counts[label] / total
        print(f"{label:<40} {counts[label]:>4} students  ({share:.1f}%)")


if __name__ == "__main__":
    main()