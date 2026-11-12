import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler

TARGET = "G3"
PRIOR_GRADES = ["G1" ,"G2"]


def add_pass_label(df: pd.DataFrame, pass_mark : int =10) -> pd.DataFrame:
    """Add a 'passed' column: 1 if G3 >= pass_mark, else 0."""
    return df.assign(passed=(df[TARGET] >= pass_mark).astype(int))

def get_feature_columns(df:pd.DataFrame,include_prior_grades:bool = True)->list :
    """Input columns for the models (never the target or the pass label)."""

    cols = [c for c in df.columns if c not in (TARGET ,"passed")]
    if not include_prior_grades:
        cols = [c for c in cols if c not in PRIOR_GRADES]

    return cols


def split_data(df:pd.DataFrame,pass_mark: int = 10, test_size:float =0.2,random_state :  int = 42 ):

    """Split students into train and test sets (same students for every model)."""

    data = add_pass_label(df,pass_mark)
    train_df , test_df = train_test_split(
        data,
        test_size=test_size,
        stratify=data["passed"],  # keep the pass rate similar in both parts

    )

    return train_df , test_df 


def build_preprossesor(X:pd.DataFrame)-> ColumnTransformer:
    """Scale numeric columns and one-hot encode text columns.

    Models only understand numbers, so 'yes'/'no' and 'GP'/'MS' become 0/1 columns.
    handle_unknown='ignore' stops the app crashing on an unseen category.
    """

    numeric_cols = X.select_dtypes(include="number").columns.to_list()
    categorical_cols = X.select_dtypes(exclude="number").columns.to_list()

    return ColumnTransformer([
        ('num',StandardScaler(),numeric_cols)
        ('cat',OneHotEncoder(handle_unknown="ignore" , sparse_output=False),categorical_cols
         ),
    ])






