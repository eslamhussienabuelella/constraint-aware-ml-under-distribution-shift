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

def evaluate_phase_performance(models_dict, X_evaluation, y_true):
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
    """
    Evaluate multiple regression models for a given experimental phase
    using MSE, RMSE, MAPE, and R².

    Parameters
    ----------
    models_dict : dict
        Dictionary of trained models.
        Example: {"LR": model1, "Lasso": model2, "RF": model3}
    X_evaluation : pd.DataFrame or array-like
        Feature matrix used for evaluation.
    y_true : pd.Series or array-like
        Ground-truth target values.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row corresponds to a model and columns
        contain evaluation metrics.
    """

    results = []

    for model_name, model in models_dict.items():

        y_pred = model.predict(X_evaluation)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        r2 = r2_score(y_true, y_pred)

        results.append({"Model": model_name, 
                        "MSE": mse, 
                        "RMSE": rmse, 
                        "MAPE (%)": mape, 
                        "R2": r2})

    return pd.DataFrame(results)

def compute_deltas(test_df, ford_df, phase_name):
    df = test_df.copy()
    df = df[['Model', 'RMSE', 'R2']].merge(
        ford_df[['Model', 'RMSE', 'R2']],
        on='Model',
        suffixes=('_test', '_ford')
    )
    
    df['ΔRMSE'] = df['RMSE_ford'] - df['RMSE_test']
    df['ΔR2'] = df['R2_ford'] - df['R2_test']
    df['Phase'] = phase_name
    
    return df[['Phase', 'Model', 'ΔRMSE', 'ΔR2']]
