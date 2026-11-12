import pandas as pd
from sklearn.dummy import DummyClassifier , DummyRegressor 
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.linear_model import LinearRegression , LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor

from aislca.data.features  import build_preprossesor

def get_regresssion_models(random_state :  int = 42) -> dict :
    """Models that predict the final mark (0 to 20)."""
    return{
        "Baseline (mean)":DummyRegressor(strategy="mean"),
        "Linear Regression":LinearRegression(),
        "Decision Tree":DecisionTreeRegressor(max_depth=4 , random_state=random_state),
        "Random Forest" : RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2,random_state=random_state
        ),
    }

def build_pipelines(models:dict , X:pd.DataFrame)->dict:
    """Wrap each model with preprocessing (not trained yet)."""

    return {
        name:Pipeline([("preprocess",build_preprossesor(X)),("model",models)])
        for name, model in models.items()
    }


def train_models(models: dict , X_train:pd.DataFrame,y_train:pd.Series)->dict:
    """Train every model on the training data. Returns the fitted pipelines."""

    pipelines  = build_preprossesor(models,X_train)
    for pipeline in pipelines.values():
        pipeline.fit(X_train,y_train)
        return pipelines

def feature_importance(pipeline,top_n: int = 10 ) -> pd.Series:
    """Most important inputs for a tree-based pipeline (Random Forest / Decision Tree)."""
    names =  pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    return pd.Series(importances,index=names).sort_values(ascending=False).head(top_n).round(3)