"""Streamlit app: Student Performance Prediction and Support (Version 2)."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Let Python find our code in the src folder (needed on Streamlit Cloud and Docker)
PROJECT_FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_FOLDER / "src"))

from aislca.analytics import eda
from aislca.data import cleaning
from aislca.data.ingestion import load_config, load_processed_data
from aislca.ml import grouping, predict
from aislca.recommendations import rules

st.set_page_config(page_title="Student Performance Prediction", page_icon="🎓", layout="wide")

ESTIMATE_NOTE = (
    "These are estimates made by a computer model from past data. "
    "They are not guarantees. Use them to support a conversation with a "
    "teacher or counselor, never to replace one."
)

STUDY_TIME_OPTIONS = {
    "Less than 2 hours a week": 1,
    "2 to 5 hours a week": 2,
    "5 to 10 hours a week": 3,
    "More than 10 hours a week": 4,
}

PRIORITY_LABELS = {
    "high": "High priority",
    "medium": "Medium priority",
    "low": "Low priority",
    "info": "Note",
}

RULE_LIMIT_LABELS = {
    "weak_mark_below": "Mark below this = needs improvement (0 to 20)",
    "moderate_mark_below": "Mark below this = moderate",
    "strong_mark_from": "Mark from this = strong",
    "low_study_time_code": "Study time code that counts as low (1 = under 2 hours a week)",
    "high_absences": "Absences in one subject that trigger advice",
    "failures_flag_from": "Earlier class failures that trigger advice",
    "big_subject_gap": "Gap between two subjects that triggers advice (marks)",
    "practice_sessions_per_week": "Suggested practice sessions per week",
    "practice_minutes": "Suggested minutes per practice session",
    "pass_mark": "Pass mark (0 to 20)",
}


# ---------- Loading (done once, then remembered) ----------

@st.cache_resource
def get_model_bundle():
    """Load the saved model file."""
    return predict.load_model_bundle()


@st.cache_resource
def get_group_bundle():
    """Load the saved student-group model. Returns None if the file is missing."""
    try:
        return grouping.load_group_bundle()
    except FileNotFoundError:
        return None


@st.cache_data
def get_students(subject):
    """Load a cleaned dataset: 'mat' for Math, 'por' for Portuguese."""
    return load_processed_data(subject)


@st.cache_data
def get_rule_settings():
    """The limits used by the recommendation rules (from the config file)."""
    return rules.load_rule_settings()


# ---------- Tab 1: predict one student ----------

def show_predict_tab(bundle):
    st.subheader("Estimate one student's result")
    st.write(
        "Answer the questions below. Any question not shown here "
        "uses a typical value from the training data."
    )

    student = {}
    left, right = st.columns(2)

    with left:
        if bundle["uses_prior_grades"]:
            student["G1"] = st.slider("First-period grade (G1, 0 to 20)", 0, 20, 10)
            student["G2"] = st.slider("Second-period grade (G2, 0 to 20)", 0, 20, 10)
        study_choice = st.selectbox(
            "Weekly study time", list(STUDY_TIME_OPTIONS.keys()), index=1
        )
        student["studytime"] = STUDY_TIME_OPTIONS[study_choice]
        student["failures"] = st.selectbox("Past class failures", [0, 1, 2, 3])

    with right:
        student["absences"] = st.number_input("Number of absences", 0, 100, 4)
        student["age"] = st.number_input("Age", 15, 22, 17)
        student["higher"] = st.selectbox("Wants to study higher education?", ["yes", "no"])
        student["goout"] = st.slider("Going out with friends (1 = very low, 5 = very high)", 1, 5, 3)

    if st.button("Get estimate"):
        result = predict.predict_one(bundle, student)
        chance_percent = round(result["pass_chance"] * 100)

        box1, box2, box3 = st.columns(3)
        box1.metric("Estimated final mark (0 to 20)", result["estimated_mark"])
        box2.metric(f"Chance of reaching the pass mark ({bundle['pass_mark']})", f"{chance_percent}%")
        box3.metric("Estimate", result["estimate"])
        st.info(ESTIMATE_NOTE)


# ---------- Tab 2: weak spots and recommendations (new in V2) ----------

def show_recommendation(item):
    """Show one recommendation with its plain-language reason."""
    label = PRIORITY_LABELS[item["priority"]]
    st.markdown(f"**{label}: {item['area']}**")
    st.write("**What to do:** " + item["action"])
    st.write("**Why:** " + item["why"])
    if item["goal"]:
        st.write("**Goal:** " + item["goal"])
    st.divider()


def show_group_box(math_student):
    """Show the student group that is closest to this student."""
    group_bundle = get_group_bundle()
    if group_bundle is None:
        st.info("The student-group model was not found. Create it with: python -m aislca.ml.grouping")
        return

    assignment = grouping.assign_group(group_bundle, math_student)
    st.markdown("**Closest group of students: " + assignment["group"] + "**")
    st.write(assignment["description"])

    if len(assignment["used_defaults"]) > 0:
        names = []
        for feature in assignment["used_defaults"]:
            names.append(grouping.FEATURE_WORDS.get(feature, feature))
        st.caption("Typical values were used for: " + ", ".join(names) + ".")

    st.caption(
        "The description is about the group as a whole, not about this student. "
        "Groups are weakly separated patterns in past data. They are not labels of "
        "people and not predictions."
    )


def show_recommendations_tab(bundle):
    st.subheader("Weak spots and what to do about them")
    st.write(
        "Enter a student's current marks and habits. The advice comes from simple, "
        "visible rules, and every item explains why. The data has no topic-level marks, "
        "so the advice is about subjects and study habits."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Mathematics**")
        g1 = st.slider("First-period Mathematics grade (G1, 0 to 20)", 0, 20, 10, key="rec_g1")
        g2 = st.slider("Current Mathematics mark (second period, G2, 0 to 20)", 0, 20, 10, key="rec_g2")
        math_absences = st.number_input("Absences in Mathematics", 0, 100, 4, key="rec_math_absences")
        math_failures = st.selectbox("Past class failures in Mathematics", [0, 1, 2, 3], key="rec_math_failures")
        study_choice = st.selectbox(
            "Weekly study time", list(STUDY_TIME_OPTIONS.keys()), index=1, key="rec_study")

    with right:
        st.markdown("**Portuguese (optional)**")
        include_portuguese = st.checkbox("Also include a Portuguese mark", key="rec_include_por")
        por_mark = None
        por_absences = 0
        por_failures = 0
        if include_portuguese:
            por_mark = st.slider("Current Portuguese mark (second period, 0 to 20)", 0, 20, 10, key="rec_por_mark")
            por_absences = st.number_input("Absences in Portuguese", 0, 100, 4, key="rec_por_absences")
            por_failures = st.selectbox("Past class failures in Portuguese", [0, 1, 2, 3], key="rec_por_failures")

    study_time = STUDY_TIME_OPTIONS[study_choice]

    # The model estimate uses the Math answers (the model was trained on Math data)
    math_student = {
        "G1": g1,
        "G2": g2,
        "studytime": study_time,
        "absences": math_absences,
        "failures": math_failures,
    }
    estimate = predict.predict_one(bundle, math_student)
    fail_probability = predict.fail_probability_one(bundle, math_student)
    pass_probability = 1 - fail_probability
    threshold = bundle.get("support_threshold", 0.5)

    st.markdown("### Model estimate for Mathematics")
    box1, box2, box3 = st.columns(3)
    box1.metric("Estimated final Mathematics mark (0 to 20)", estimate["estimated_mark"])
    box2.metric(f"Chance of reaching the pass mark ({bundle['pass_mark']})", f"{round(pass_probability * 100)}%")
    box3.metric("Chance of not reaching it", f"{round(fail_probability * 100)}%")

    if fail_probability >= threshold:
        st.warning(
            f"The estimated chance of not passing is at or above the flag threshold "
            f"({round(threshold * 100)}%), so extra support may be worth discussing. "
            "This is an estimate, not a certainty."
        )
    else:
        st.info(
            f"The estimated chance of not passing is below the flag threshold "
            f"({round(threshold * 100)}%). That does not mean there is nothing to work on. "
            "See the advice below."
        )

    # The rules use the current marks (G2) in each subject
    marks = {"Mathematics": g2}
    absences = {"Mathematics": math_absences}
    past_failures = {"Mathematics": math_failures}
    if include_portuguese:
        marks["Portuguese"] = por_mark
        absences["Portuguese"] = por_absences
        past_failures["Portuguese"] = por_failures

    rule_student = {
        "marks": marks,
        "study_time": study_time,
        "absences": absences,
        "past_failures": past_failures,
        "fail_probability": fail_probability,
    }
    result = rules.explain_student(rule_student, get_rule_settings(), support_threshold=threshold)

    st.markdown("### Subjects")
    table = pd.DataFrame(result["subjects"])
    table.columns = ["Subject", "Current mark", "Performance"]
    st.table(table)

    st.markdown("### Advice")
    for item in result["recommendations"]:
        show_recommendation(item)

    st.markdown("### Student group")
    show_group_box(math_student)

    st.warning(ESTIMATE_NOTE)
    st.caption(result["disclaimer"])


# ---------- Tab 3: explore the data ----------

def show_explore_tab():
    st.subheader("Explore the data")
    students = get_students("mat")
    pass_mark = load_config()["target"]["pass_mark"]

    if st.checkbox("Show the first rows of the data"):
        st.dataframe(students.head(20))

    chart_names = [
        "Grade distribution",
        "Absences vs final grade",
        "Average grade by study time",
        "Average grade by past failures",
        "Correlation heatmap",
        "Math vs Portuguese averages",
    ]
    choice = st.selectbox("Choose a chart", chart_names)

    if choice == "Grade distribution":
        figure = eda.plot_grade_distribution(students, pass_mark=pass_mark)
    elif choice == "Absences vs final grade":
        figure = eda.plot_scatter_vs_grade(students, "absences")
    elif choice == "Average grade by study time":
        figure = eda.plot_mean_grade_by(students, "studytime")
    elif choice == "Average grade by past failures":
        figure = eda.plot_mean_grade_by(students, "failures")
    elif choice == "Correlation heatmap":
        figure = eda.plot_correlation_heatmap(students)
    else:
        figure = eda.plot_subject_averages(students, get_students("por"))

    st.pyplot(figure)
    st.caption(
        "Data: UCI Student Performance (Math, 395 students). "
        "A pattern in a chart does not prove a cause."
    )


# ---------- Tab 4: upload a CSV ----------

def show_upload_tab(bundle):
    st.subheader("Upload a CSV file of students")
    st.warning(
        "Only upload public or practice data, for example the UCI files. "
        "Do not upload real student records."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is None:
        st.write("Waiting for a file. You can try data/processed/student_mat_clean.csv from this project.")
        return

    # Read and clean the file with the same code we wrote in Task 3
    try:
        raw_students = pd.read_csv(uploaded_file, sep=None, engine="python")
        clean_students, report = cleaning.clean_student_data(raw_students)
    except Exception as error:
        st.error(f"Could not read this file: {error}")
        return

    st.write(f"Rows read: {report['rows_before']}. Rows kept after cleaning: {report['rows_after']}.")
    if len(clean_students) == 0:
        st.error("No valid rows were left after cleaning.")
        return
    st.dataframe(clean_students.head(10))

    # Tell the user which inputs were missing from their file
    missing_columns = []
    for column in bundle["feature_columns"]:
        if column not in clean_students.columns:
            missing_columns.append(column)
    if len(missing_columns) > 0:
        st.warning("These columns are not in your file, so typical values are used: " + ", ".join(missing_columns))

    # Predict for every student
    results = predict.predict_students(bundle, clean_students)
    st.subheader("Estimates")
    st.dataframe(results)
    st.download_button(
        "Download estimates as CSV",
        data=results.to_csv(index=False),
        file_name="student_estimates.csv",
        mime="text/csv",
    )
    st.caption(ESTIMATE_NOTE)

    # Show a chart if the file has final grades
    if "G3" in clean_students.columns:
        st.subheader("Final grades in your file")
        figure = eda.plot_grade_distribution(clean_students, pass_mark=bundle["pass_mark"])
        st.pyplot(figure)


# ---------- Tab 5: about the model ----------

def show_model_comparison(metrics):
    """Version 2: the cross-validation tables for every model that was tried."""
    st.write("**Model comparison** (cross-validation on the training students only)")
    st.write(
        "A model has to beat another by more than its spread (std) to count as better. "
        "Smaller gaps are ties."
    )
    st.write("Final mark models (RMSE in grade points, lower is better):")
    st.table(pd.DataFrame(metrics["regression"]["cross_validation"]).T)
    st.write("Pass/fail models (F1 of the pass group, higher is better):")
    st.table(pd.DataFrame(metrics["classification"]["cross_validation"]).T)
    st.caption(
        "XGBoost and tuned versions of each model were tried. "
        "The final models listed above were chosen by this comparison."
    )


def show_support_analysis(classification):
    """Version 2: how well we find students who may need support."""
    support = classification.get("support_analysis")
    if support is None:
        return

    chosen = support["chosen_threshold"]
    st.write("**Finding students who may need support**")
    st.write(
        "A student is flagged when the estimated chance of not passing is at least "
        f"{round(chosen * 100)}%. This threshold was chosen on the training students only: "
        f"the highest one that still found at least {round(support['target_recall'] * 100)}% "
        "of the failing students."
    )
    st.write("Test students, at the old threshold (50%) and at the chosen one:")
    st.table(pd.DataFrame({
        "Threshold 50%": support["test_at_default_threshold"],
        f"Threshold {round(chosen * 100)}%": support["test_at_chosen_threshold"],
    }))

    counts = classification["confusion_matrix"]["counts"]
    failed = counts[0][0] + counts[0][1]
    if failed > 0:
        st.caption(
            f"Only {failed} failing students are in the test set, so one student moves "
            f"recall by about {100 / failed:.1f} percentage points. Small differences are noise."
        )

    st.write("Fail-group comparison of the models (training students, cross-validated):")
    st.table(pd.DataFrame(support["model_comparison_training"]).T)
    st.write("Threshold analysis (training students, cross-validated):")
    st.dataframe(pd.DataFrame(support["threshold_table_training"]))


def show_group_report(config):
    """Version 2: the student groups."""
    report_file = PROJECT_FOLDER / config["groups"]["report_path"]
    if not report_file.exists():
        return

    with open(report_file, encoding="utf-8") as file:
        report = json.load(file)

    chosen_k = report["chosen_k"]
    chosen_score = None
    for row in report["silhouette_by_k"]:
        if row["k"] == chosen_k:
            chosen_score = row["silhouette"]

    st.write("**Student groups**")
    st.write(
        f"Students were split into {chosen_k} groups by study time, absences, past failures "
        f"and period grades. The number of groups was chosen with the silhouette score "
        f"({chosen_score})."
    )
    st.caption(
        "Silhouette: 1 = very clear groups, 0 = overlapping groups. Values below about 0.5 "
        "mean the groups overlap, so treat them as descriptive slices, not natural clusters."
    )
    for group in report["groups"]:
        st.write(f"**{group['name']}** ({group['students']} students): {group['description']}")


def show_rule_limits():
    """Version 2: the limits used by the recommendation rules."""
    settings = get_rule_settings()
    names = []
    values = []
    for key, value in settings.items():
        names.append(RULE_LIMIT_LABELS.get(key, key))
        values.append(str(value))

    st.write("**Limits used by the recommendation rules**")
    st.table(pd.DataFrame({"Setting": names, "Value": values}))
    st.caption(
        "These are starting values chosen for this project. They have not been validated "
        "by teachers. They can be changed in configs/config.yaml."
    )


def show_about_tab():
    st.subheader("About this model")

    config = load_config()
    metrics_file = PROJECT_FOLDER / config["registry"]["metrics_path"]
    if not metrics_file.exists():
        st.write("No metrics file found. Run: python -m aislca.ml.train")
        return

    with open(metrics_file, encoding="utf-8") as file:
        metrics = json.load(file)

    st.write(
        f"Trained on {metrics['train_students']} students and tested on "
        f"{metrics['test_students']} different students."
    )

    regression = metrics["regression"]
    st.write("**Final mark model:** " + regression["best_model"])
    st.write("MAE and RMSE are in grade points (lower is better). R2 closer to 1 is better.")
    st.table(pd.DataFrame({
        "Our model": regression["test_scores"],
        "Guessing the average": regression["baseline_test_scores"],
    }))

    classification = metrics["classification"]
    st.write("**Pass/fail model:** " + classification["best_model"])
    st.table(pd.DataFrame({
        "Pass group": classification["test_scores_pass_group"],
        "Fail group": classification["test_scores_fail_group"],
        "Always saying pass": classification["baseline_test_scores"],
    }))

    matrix = classification["confusion_matrix"]
    st.write("Confusion matrix (test students):")
    st.table(pd.DataFrame(matrix["counts"], index=matrix["rows"], columns=matrix["columns"]))

    st.subheader("Version 2 results")
    show_model_comparison(metrics)
    show_support_analysis(classification)
    show_group_report(config)
    show_rule_limits()

    st.subheader("Responsible use")
    st.write("- Predictions depend on the quality and amount of data.")
    st.write("- This model learned from two Portuguese schools and may not work for other students.")
    st.write("- Estimates support teachers and students. They do not replace them.")
    st.write("- Recommendations come from simple rules with limits that teachers should review.")
    st.write("- Student groups are weakly separated patterns, not labels of people.")
    st.write("- The model relies on earlier period grades (G1 and G2), so it confirms low grades more than it warns early.")
    st.write("- Data source: UCI Student Performance dataset (public).")


# ---------- Put the page together ----------

def main():
    st.title("🎓 Student Performance Prediction")
    st.caption("Version 2 of the AI-Powered Smart Learning & Career Assistant")
    st.warning(ESTIMATE_NOTE)

    try:
        bundle = get_model_bundle()
    except FileNotFoundError:
        st.error("Model file not found. Create it with: python -m aislca.ml.train")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Predict", "Recommendations", "Explore data", "Upload CSV", "About"]
    )
    with tab1:
        show_predict_tab(bundle)
    with tab2:
        show_recommendations_tab(bundle)
    with tab3:
        show_explore_tab()
    with tab4:
        show_upload_tab(bundle)
    with tab5:
        show_about_tab()


main()