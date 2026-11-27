"""Evaluation for the goal "find students who may need support" (V2).

A student predicted to FAIL (final mark below the pass mark) is flagged
as "may need support". We judge the flags by:
  recall    = of the students who really fail, how many did we flag?
  precision = of the students we flag, how many really fail?
Everything here uses plain Python numbers so it saves cleanly as JSON.
"""

from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# V1 flagged a student when the fail probability was 0.5 or more
DEFAULT_THRESHOLD = 0.5


def fail_group_scores(y_passed, flagged):
    """Count the flags and score them.

    y_passed: 1 = student passed, 0 = student failed.
    flagged:  True = we flagged the student as "may need support".
    """
    caught = 0         # failing students we flagged
    missed = 0         # failing students we did not flag
    false_alarms = 0   # passing students we flagged

    for passed, is_flagged in zip(y_passed, flagged):
        if passed == 0 and is_flagged:
            caught += 1
        elif passed == 0:
            missed += 1
        elif is_flagged:
            false_alarms += 1

    total_flagged = caught + false_alarms
    total_failed = caught + missed

    if total_failed > 0:
        recall = caught / total_failed
    else:
        recall = 0.0

    if total_flagged > 0:
        precision = caught / total_flagged
    else:
        precision = 0.0

    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0.0

    return {
        "flagged": total_flagged,
        "caught": caught,
        "missed": missed,
        "false_alarms": false_alarms,
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
    }


def get_fail_probability(model, X):
    """The model's estimated probability that each student fails."""
    probabilities = model.predict_proba(X)
    fail_column = list(model.classes_).index(0)   # class 0 = fail
    return probabilities[:, fail_column]


def out_of_fold_fail_probability(model, X, y, cv_folds, random_state):
    """Fail probability for each training student from a model that never saw them.

    The data is split into folds. Each student is scored by a model trained
    on the other folds. This gives honest probabilities without the test set.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    probabilities = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return probabilities[:, 0]   # column 0 = class 0 = fail


def make_thresholds():
    """The thresholds to try: 0.10, 0.15, ... 0.80."""
    thresholds = []
    value = 0.10
    while value <= 0.80001:
        thresholds.append(round(value, 2))
        value = value + 0.05
    return thresholds


def threshold_table(y_passed, fail_probabilities, thresholds):
    """Score the flags at every threshold. Returns a list of rows."""
    rows = []
    for threshold in thresholds:
        flagged = []
        for probability in fail_probabilities:
            flagged.append(probability >= threshold)

        scores = fail_group_scores(y_passed, flagged)
        row = {"threshold": threshold}
        for key, value in scores.items():
            row[key] = value
        rows.append(row)
    return rows


def pick_threshold(rows, target_recall):
    """The highest threshold that still reaches the target recall.

    Higher thresholds flag fewer students (fewer false alarms), so we go as
    high as we can while still finding enough of the failing students.
    If no threshold reaches the target, we keep the default of 0.5.
    """
    chosen = None
    for row in rows:   # rows go from the lowest threshold to the highest
        if row["recall"] >= target_recall:
            chosen = row["threshold"]

    if chosen is None:
        return DEFAULT_THRESHOLD
    return chosen


def compare_models_for_support(models, X, y, cv_folds, random_state):
    """Compare models on the fail group, using training students only.

    roc_auc measures how well a model ranks students from most to least likely
    to fail, whatever the threshold: 0.5 = guessing, 1.0 = perfect ranking.
    """
    actual_fail = []
    for passed in y:
        if passed == 0:
            actual_fail.append(1)
        else:
            actual_fail.append(0)

    results = {}
    for name, model in models.items():
        if "Baseline" in name:
            continue

        fail_probabilities = out_of_fold_fail_probability(
            clone(model), X, y, cv_folds, random_state)

        flagged = []
        for probability in fail_probabilities:
            flagged.append(probability >= DEFAULT_THRESHOLD)

        scores = fail_group_scores(y, flagged)
        scores["roc_auc"] = round(float(roc_auc_score(actual_fail, fail_probabilities)), 3)
        results[name] = scores
    return results


def analyse_support(final_model, candidate_models, X_train, y_train,
                    X_test, y_test, target_recall, cv_folds, random_state):
    """The full "who may need support?" analysis.

    1. Compare the candidate models (training students only).
    2. Build the threshold table for the final model (training students only).
    3. Choose the threshold (training students only).
    4. Score the test students ONCE at 0.5 and once at the chosen threshold.
    """
    comparison = compare_models_for_support(
        candidate_models, X_train, y_train, cv_folds, random_state)

    train_fail_probabilities = out_of_fold_fail_probability(
        clone(final_model), X_train, y_train, cv_folds, random_state)
    rows = threshold_table(y_train, train_fail_probabilities, make_thresholds())
    chosen = pick_threshold(rows, target_recall)

    test_fail_probabilities = get_fail_probability(final_model, X_test)
    flagged_default = []
    flagged_chosen = []
    for probability in test_fail_probabilities:
        flagged_default.append(probability >= DEFAULT_THRESHOLD)
        flagged_chosen.append(probability >= chosen)

    return {
        "target_recall": target_recall,
        "default_threshold": DEFAULT_THRESHOLD,
        "chosen_threshold": chosen,
        "model_comparison_training": comparison,
        "threshold_table_training": rows,
        "test_at_default_threshold": fail_group_scores(y_test, flagged_default),
        "test_at_chosen_threshold": fail_group_scores(y_test, flagged_chosen),
    }