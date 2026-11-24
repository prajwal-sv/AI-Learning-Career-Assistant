# AI-Powered Smart Learning & Career Assistant (AISLCA)

![CI](https://github.com/YOUR-USERNAME/AI-Learning-Career-Assistant/actions/workflows/ci.yml/badge.svg)

A course-long AI project, built step by step in Python. Every version is a
live, deployable app.

**Current version: V1 - Student Performance Prediction System**

- **Live app:** TODO add the Streamlit link after deployment
- **Code:** https://github.com/YOUR-USERNAME/AI-Learning-Career-Assistant

![App screenshot](docs/screenshot.png)

---

## What V1 does

V1 uses a public student dataset to:

- clean and explore student data,
- show which factors relate to final marks,
- group similar students (K-Means clustering),
- predict a student's final mark and chance of passing,
- let anyone try it in a web app.

**Predictions are estimates, not guarantees.** See "Responsible use" below.

## The web app

The Streamlit app has four tabs:

| Tab | What you can do |
|---|---|
| **Predict** | Answer a few questions and get an estimated final mark (0 to 20) and a pass chance |
| **Explore data** | View charts: grade distribution, absences, study time, past failures, correlation heatmap, Math vs Portuguese |
| **Upload CSV** | Upload a file of students, get an estimate for each one, and download the results |
| **About** | See the model scores, the confusion matrix, and responsible-use notes |

## Data

- **Dataset:** UCI Student Performance (public, CC BY 4.0 license).
- **Source:** https://archive.ics.uci.edu/dataset/320/student+performance
- **Content:** 395 Math students and 649 Portuguese students from two
  Portuguese secondary schools, with 33 columns each.
- **Target:** `G3` (final grade, 0 to 20). A student passes if `G3 >= 10`.
- **Note:** the dataset has no attendance percentage and no quiz marks.
  `absences` is used as the attendance signal, and `G1` and `G2` are the
  earlier grades.
- No real or private student records are used anywhere in this project.

Citation: P. Cortez and A. Silva. *Using Data Mining to Predict Secondary
School Student Performance.* In A. Brito and J. Teixeira (Eds.), Proceedings
of the 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12,
Porto, Portugal, April 2008, EUROSIS.

## Methods

1. **Cleaning:** missing values, duplicates, types, impossible values and an
   outlier report.
2. **Exploration:** charts and correlation analysis.
3. **Statistics:** correlation tests, t-tests, ANOVA and chi-square, with
   effect sizes.
4. **Models:** Linear Regression, Logistic Regression, Decision Tree and
   Random Forest, each compared against a simple baseline.
5. **Clustering:** K-Means on study habits and earlier grades.
6. **Evaluation:** MAE, RMSE, R2, accuracy, precision, recall, F1 and a
   confusion matrix.

The best models are chosen with cross-validation on the training data only,
and then scored once on the held-out test students.

## Results

Run `cat models/registry/metrics.json` to see the saved scores, then fill in
this table.

| Question | Best model | Test scores | Baseline (just guessing) |
|---|---|---|---|
| Final mark (0 to 20) | TODO | MAE TODO, RMSE TODO, R2 TODO | TODO |
| Pass or fail | TODO | Accuracy TODO, F1 TODO | TODO |

**Important:** the model uses the earlier grades `G1` and `G2`, which are only
known partway through the year. Without them, predictions are much weaker.

## Run it on your computer

You need Python (this project was built with Python 3.14, so use the same
version) and Git.

```bash
# 1. Get the code
git clone https://github.com/YOUR-USERNAME/AI-Learning-Career-Assistant.git
cd AI-Learning-Career-Assistant

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # Mac / Linux

# 3. Install the packages
pip install -r requirements.txt
pip install -e .

# 4. Start the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Re-create the data and the model (optional)

The cleaned data and the trained model are already included. To rebuild them:

```bash
python -m aislca.data.cleaning     # cleans the raw data
python -m aislca.ml.train          # trains and saves the model and metrics
```

### Run the tests

```bash
python -m pytest tests -v
```

## Run with Docker

```bash
docker build -t aislca .
docker run -p 8501:8501 aislca
```

Then open http://localhost:8501.

## Project structure

```
AI-Learning-Career-Assistant/
├── app.py                  Streamlit app
├── configs/config.yaml     Settings (pass mark, split, clustering, model paths)
├── data/
│   ├── raw/                Original dataset (never edited)
│   └── processed/          Cleaned data
├── notebooks/              Exploration (01 data analysis, 03 machine learning)
├── src/aislca/
│   ├── data/               ingestion, cleaning, features
│   ├── analytics/          eda (charts), stat_tests
│   └── ml/                 train, evaluate, predict, clustering
├── models/registry/        Saved model and metrics.json
├── tests/unit/             Unit tests
├── Dockerfile
└── .github/workflows/      Automatic tests (GitHub Actions)
```

Reusable logic lives in `src/`. Notebooks are only for exploring.

## Responsible use

- Predictions depend on the quality and amount of data.
- The model learned from two Portuguese schools and may not work for other
  schools, countries or years.
- Predictions are estimates, not guarantees. They should support teachers and
  students, never replace human judgment.
- Student clusters describe patterns. They are not labels or judgments about a
  person's ability.
- Do not upload real student records to the app.
- Some columns (such as sex or alcohol use) are sensitive. A real school tool
  would need to be tested for bias and unfair outcomes before use.

## Limitations

- Small dataset (395 Math students), so scores change a little with a
  different random split.
- Students who scored 0 in the final grade (likely missed exams) are hard to
  predict from the available data.
- Correlation in the charts does not prove cause.

## Roadmap

| Version | Focus |
|---|---|
| **V1** | Data analytics and ML prediction (this version) |
| V2 | Advanced ML and recommendations |
| V3 | Deep Learning models |
| V4 | Resume analyzer and NLP chatbot |
| V5 | Computer Vision and OCR notes scanner |
| V6 | Generative AI and RAG |
| V7 | Cloud deployment and complete application |

## Tech stack

Python, Pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Streamlit,
pytest, Docker and GitHub Actions. Everything used is free.