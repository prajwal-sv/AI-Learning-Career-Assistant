import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from aislca.data.features import get_feature_columns, prepare_inputs, split_data
from aislca.data.ingestion import PROJECT_ROOT, load_config, load_processed_data
from aislca.ml import evaluate

def get_regression_models(random_state = 42) :
    """Models that predict the final mark (0 to 20)."""
    models= {
        "Baseline (mean)":DummyRegressor(strategy="mean"),
        "Linear Regression":LinearRegression(),
        "Decision Tree":DecisionTreeRegressor(max_depth=4 , random_state=random_state),
        "Random Forest" : RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2,random_state=random_state
        ),
        "XGBoost" : XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=random_state, n_jobs=1,
        )
    }
    return models

def get_classification_models(random_state = 42) :
    """Models that predict pass (1) or fail (0)."""
    models =  {
        "Baseline (always pass)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, random_state=random_state
        ),
         "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=random_state, n_jobs=1,
        ),
    }
    return models


def train_models(models, X_train,y_train):
    """Train every model on the training data. Returns the fitted pipelines."""

    train_models ={}
    for name , model in models.items():
        model.fit(X_train,y_train)
        train_models[name] = model

    return train_models
    

def feature_importances(model,column_names,top_n=10 ) :
    """Show which inputs a tree-based model used most."""
    importances = pd.Series(model.feature_importances_, index=column_names)
    return importances.sort_values(ascending=False).head(top_n).round(3)


def get_default_values(students, feature_columns):
    """A typical value for every input column.

    Numbers get the middle value (median). Text gets the most common answer.
    The app uses these for any question the user does not answer.
    """
    defaults = {}
    for column in feature_columns:
        if pd.api.types.is_numeric_dtype(students[column]):
            defaults[column] = float(students[column].median())
        else:
            defaults[column] = str(students[column].mode()[0])
    return defaults



def run_final_training():
    """Choose the best models, test them once, and save everything."""
    config = load_config()
    pass_mark = config["target"]["pass_mark"]
    test_size = config["ml"]["test_size"]
    random_state = config["ml"]["random_state"]
    cv_folds = config["ml"]["cv_folds"]
    use_prior_grades = config["registry"]["include_prior_grades"]

    # Step 1: load the cleaned data and split it into train and test students
    students = load_processed_data("mat", config)
    train_data, test_data = split_data(students, pass_mark, test_size, random_state)

    # Step 2: turn the inputs into numbers
    feature_columns = get_feature_columns(students, include_prior_grades=use_prior_grades)
    X_train = prepare_inputs(train_data, feature_columns)
    X_test = prepare_inputs(test_data, feature_columns, list(X_train.columns))

    train_marks = train_data["G3"]
    test_marks = test_data["G3"]
    train_passed = train_data["passed"]
    test_passed = test_data["passed"]

    # Step 3: choose the best models using the TRAINING data only
    regression_cv = evaluate.cross_validated_scores(
        get_regression_models(random_state), X_train, train_marks,
        "regression", cv_folds, random_state)
    classification_cv = evaluate.cross_validated_scores(
        get_classification_models(random_state), X_train, train_passed,
        "classification", cv_folds, random_state)

    best_regression_name = evaluate.pick_best_model(regression_cv, "regression")
    best_classification_name = evaluate.pick_best_model(classification_cv, "classification")

    # Step 4: train the two winners
    regression_model = get_regression_models(random_state)[best_regression_name]
    regression_model.fit(X_train, train_marks)

    classification_model = get_classification_models(random_state)[best_classification_name]
    classification_model.fit(X_train, train_passed)

    # Step 5: train the baselines too, so we can show the winners beat guessing
    baseline_regression = get_regression_models(random_state)["Baseline (mean)"]
    baseline_regression.fit(X_train, train_marks)
    baseline_classification = get_classification_models(random_state)["Baseline (always pass)"]
    baseline_classification.fit(X_train, train_passed)

    # Step 6: score everything ONCE on the test students
    regression_predictions = np.clip(regression_model.predict(X_test), 0, 20)
    baseline_marks = np.clip(baseline_regression.predict(X_test), 0, 20)
    classification_predictions = classification_model.predict(X_test)
    baseline_passed = baseline_classification.predict(X_test)
    confusion = evaluate.confusion_matrix_table(test_passed, classification_predictions)

    metrics = {
        "dataset": "UCI Student Performance (Math)",
        "uses_prior_grades": bool(use_prior_grades),
        "train_students": len(train_data),
        "test_students": len(test_data),
        "regression": {
            "best_model": best_regression_name,
            "test_scores": evaluate.regression_metrics(test_marks, regression_predictions),
            "baseline_test_scores": evaluate.regression_metrics(test_marks, baseline_marks),
            "cross_validation": regression_cv.to_dict(orient="index"),
        },
        "classification": {
            "best_model": best_classification_name,
            "test_scores_pass_group": evaluate.classification_metrics(
                test_passed, classification_predictions, pos_label=1),
            "test_scores_fail_group": evaluate.classification_metrics(
                test_passed, classification_predictions, pos_label=0),
            "baseline_test_scores": evaluate.classification_metrics(
                test_passed, baseline_passed, pos_label=1),
            "confusion_matrix": {
                "rows": list(confusion.index),
                "columns": list(confusion.columns),
                "counts": confusion.values.tolist(),
            },
            "cross_validation": classification_cv.to_dict(orient="index"),
        },
        "note": "Predictions are estimates, not guarantees.",
    }

    # Step 7: save everything the app needs in ONE file
    bundle = {
        "feature_columns": feature_columns,
        "training_columns": list(X_train.columns),
        "defaults": get_default_values(students, feature_columns),
        "pass_mark": pass_mark,
        "uses_prior_grades": bool(use_prior_grades),
        "regression_model_name": best_regression_name,
        "regression_model": regression_model,
        "classification_model_name": best_classification_name,
        "classification_model": classification_model,
    }

    model_path = PROJECT_ROOT / config["registry"]["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)

    metrics_path = PROJECT_ROOT / config["registry"]["metrics_path"]
    evaluate.save_metrics_json(metrics, metrics_path)

    print("Best mark model:     ", best_regression_name)
    print("Best pass/fail model:", best_classification_name)
    print("Model saved to:      ", model_path)
    print("Metrics saved to:    ", metrics_path)
    return metrics


if __name__ == "__main__":
    run_final_training()