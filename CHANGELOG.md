# Changelog

All notable changes to this project are listed here.

## [1.0.0] - 2026-11-26

First live release: Student Performance Prediction System (V1).

Live app: https://aislca-student-performance.streamlit.app/

### Added
- Data ingestion, cleaning and validation for the UCI Student Performance data.
- Exploratory charts and statistical tests (correlation, t-test, ANOVA, chi-square).
- Models for the final mark (Linear Regression, Decision Tree, Random Forest)
  and for pass/fail (Logistic Regression, Decision Tree, Random Forest), each
  compared against a baseline.
- Model choice by 5-fold cross-validation, final scoring on held-out test students,
  and saved model plus metrics (`models/registry/`).
- K-Means clustering of students.
- Streamlit app with four tabs: Predict, Explore data, Upload CSV, About.
- 25 unit tests and a GitHub Actions workflow that runs them on every push.
- Dockerfile and .dockerignore.
- Deployment on Streamlit Community Cloud.
- README with file guide, scoring explanation, lessons learned and responsible use.

### Fixed
- `pick_best_model` now reads the first score column, so it works with any column name.
- CI uses the same Python version as the local environment.

## [Unreleased]
- Nothing yet. V2 will add advanced ML and recommendations.