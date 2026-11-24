"""Streamlit app: Student Performance Prediction (Version 1)."""

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
from aislca.ml import predict

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


# ---------- Loading (done once, then remembered) ----------

@st.cache_resource
def get_model_bundle():
    """Load the saved model file."""
    return predict.load_model_bundle()


@st.cache_data
def get_students(subject):
    """Load a cleaned dataset: 'mat' for Math, 'por' for Portuguese."""
    return load_processed_data(subject)


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
        chance_percent = int(result["pass_chance"] * 100)

        box1, box2, box3 = st.columns(3)
        box1.metric("Estimated final mark (0 to 20)", result["estimated_mark"])
        box2.metric(f"Chance of reaching the pass mark ({bundle['pass_mark']})", f"{chance_percent}%")
        box3.metric("Estimate", result["estimate"])
        st.info(ESTIMATE_NOTE)


# ---------- Tab 2: explore the data ----------

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


# ---------- Tab 3: upload a CSV ----------

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


# ---------- Tab 4: about the model ----------

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

    st.subheader("Responsible use")
    st.write("- Predictions depend on the quality and amount of data.")
    st.write("- This model learned from two Portuguese schools and may not work for other students.")
    st.write("- Estimates support teachers and students. They do not replace them.")
    st.write("- Data source: UCI Student Performance dataset (public).")


# ---------- Put the page together ----------

def main():
    st.title("🎓 Student Performance Prediction")
    st.caption("Version 1 of the AI-Powered Smart Learning & Career Assistant")
    st.warning(ESTIMATE_NOTE)

    try:
        bundle = get_model_bundle()
    except FileNotFoundError:
        st.error("Model file not found. Create it with: python -m aislca.ml.train")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Explore data", "Upload CSV", "About"])
    with tab1:
        show_predict_tab(bundle)
    with tab2:
        show_explore_tab()
    with tab3:
        show_upload_tab(bundle)
    with tab4:
        show_about_tab()


main()