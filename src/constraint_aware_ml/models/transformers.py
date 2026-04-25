from __future__ import annotations
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None
try:
    import shap
except Exception:
    shap = None
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
try:
    from category_encoders import BinaryEncoder
except Exception:
    BinaryEncoder = None

def feature_transformer(X_df):
    """ColumnTransformer that applies RobustScaler to numerical features and BinaryEncoder to categorical features."""
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler
    from category_encoders import BinaryEncoder

    num_cols = X_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[('num', RobustScaler(), num_cols),
                                                   ('bin', BinaryEncoder(), cat_cols)])

    return preprocessor

def feature_transformer_constraint(X_df, constrained_features):
    """Constrained ColumnTransformer that applies RobustScaler to numerical features and BinaryEncoder to categorical features."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import RobustScaler
    from category_encoders import BinaryEncoder

    num_cols = X_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()

    num_constrained = [col for col in num_cols if col in constrained_features]
    num_unconstrained = [col for col in num_cols if col not in constrained_features]

    preprocessor = ColumnTransformer(transformers=[('num_constrained', RobustScaler(), num_constrained),
                                                   ('num_unconstrained', RobustScaler(), num_unconstrained),
                                                   ('bin', BinaryEncoder(), cat_cols)])

    return preprocessor
