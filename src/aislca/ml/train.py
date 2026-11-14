import pandas as pd
from sklearn.dummy import DummyClassifier , DummyRegressor 
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.linear_model import LinearRegression , LogisticRegression

from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor



def get_regression_models(random_state = 42) :
    """Models that predict the final mark (0 to 20)."""
    models= {
        "Baseline (mean)":DummyRegressor(strategy="mean"),
        "Linear Regression":LinearRegression(),
        "Decision Tree":DecisionTreeRegressor(max_depth=4 , random_state=random_state),
        "Random Forest" : RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2,random_state=random_state
        ),
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
    