# Progress Log

## V1: Student Performance Prediction System (v1.0.0, live)

- [x] Task 1: repo, environment, first push
- [x] Task 2: load dataset and explain features
- [x] Task 3: data cleaning
- [x] Task 4: EDA and visualizations
- [x] Task 5: statistical and correlation analysis
- [x] Task 6: train and compare models
- [x] Task 7: K-Means clustering
- [x] Task 8: evaluation, saved model and metrics
- [x] Task 9: Streamlit app
- [x] Task 10: unit tests and CI
- [x] Deployment: Docker, Streamlit Community Cloud, v1.0.0 release

Live app: https://aislca-student-performance.streamlit.app/
Repo: https://github.com/prajwal-sv/AI-Learning-Career-Assistant

---

## Session summary (to take to the planning chat for the V2 prompt)

### What was completed
Everything in V1: cleaning, EDA, statistics, model comparison, K-Means,
saved model with metrics, Streamlit app (live), 25 unit tests, GitHub Actions
CI, Dockerfile, README, CHANGELOG, and the v1.0.0 tag.

### Key decisions
- Dataset: UCI Student Performance, Math file (395 students) as the main data;
  Portuguese file used for the subject comparison.
- Target: `G3` (0 to 20). Pass mark: `G3 >= 10`, set in `configs/config.yaml`.
- The dataset has no attendance % and no assignment/quiz marks, so `absences`
  stands in for attendance and `G1`/`G2` for earlier marks.
- Two feature sets: A (with G1, G2) is used by the saved model and the app; B
  (without them) is much weaker.
- Split: 80/20, stratified on pass/fail, seed 42 (316 train, 79 test).
- Model choice: 5-fold cross-validation on the training students only; baselines
  can never win; the test students are scored once.
- Settings: Decision Tree max_depth 4; Random Forest 200 trees, min_samples_leaf 2;
  pass chance threshold 0.5; predicted marks clipped to 0 to 20.
- G3 = 0 students and absence outliers were kept (tested separately).
- K-Means columns: studytime, absences, failures, G1, G2 (G3 not used); k = TODO.
- Code style: very simple, beginner-level (no scikit-learn Pipeline).
- Versions pinned with `==` in `requirements.txt`; local Python 3.14.

### Results (from models/registry/metrics.json)
- Mark model: TODO. Test MAE TODO, RMSE TODO, R2 TODO (baseline RMSE TODO).
- Pass/fail model: TODO. Test accuracy TODO, F1 TODO; fail-group recall TODO
  (baseline accuracy TODO).
- Confusion matrix: see README.
- Main finding: earlier marks (G1, G2) drive the prediction; without them the
  models are only slightly better than guessing.

### File changes (V1)
- Root: app.py, requirements.txt, pyproject.toml, Dockerfile, .dockerignore,
  .gitignore, README.md, CHANGELOG.md, PROGRESS.md
- configs/config.yaml
- data/raw (UCI files), data/processed (cleaned CSVs)
- notebooks/01_data_analysis.ipynb, notebooks/03_machine_learning.ipynb
- src/aislca/data: ingestion.py, cleaning.py, features.py
- src/aislca/analytics: eda.py, stat_tests.py
- src/aislca/ml: train.py, evaluate.py, predict.py, clustering.py
- models/registry: student_model.joblib, metrics.json
- tests/unit: test_cleaning.py, test_features.py, test_metrics.py, test_predict.py
- .github/workflows/ci.yml
- docs/screenshot.png

### Problems solved
- Dataset differed from the plan (no attendance %, no quiz marks).
- G1/G2 dominate the target, so two feature sets were reported.
- Tiny test set, so cross-validation was used for choosing models.
- Simplified the code (removed Pipeline/ColumnTransformer).
- Model file needs matching package versions, so versions were pinned.
- A test caught a column-name mismatch in `pick_best_model` (fixed).
- CI was red because the workflow used Python 3.11 while packages were pinned
  on 3.14; also a stray `[200~` paste marker stopped a `git add`.
- Test files were first not pushed (add, commit, push fixed it).

### Next step
Start V2: advanced ML (XGBoost, hyperparameter tuning) and recommendations.
Ideas to discuss in the planning chat:
- Rule-based weakness detection and study recommendations (threshold must be
  configurable: the project's "below 50" rule is "below 10" on a 0 to 20 scale).
- Per-subject weakness needs per-subject marks. The Math and Portuguese files
  overlap (the UCI download includes an R merge script), so a merged dataset
  could give each student marks in two subjects.
- Add new folders only when needed (for example a recommendations module),
  keep one repo, and keep every version live.