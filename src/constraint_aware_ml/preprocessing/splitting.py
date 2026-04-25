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

def ml_data_splitter(df, target_cols, drop_cols, holdout_make, make_col):
    """
    Perform data splitting and feature selection for multi-target regression proposed for this study.

    Workflow:
    1. Manufacturer hold-out (e.g., Ford) → Out-of-Distribution (OOD) test set
    2. Remaining data split into:
        - 60% training
        - 20% validation
        - 20% in-distribution test
    3. Targets are separated and leakage-prone identifiers are removed after data splitting to prevent data leakage while minimising code redundancy 
    4. Training + validation combined to form 80% final training set to use after model fine tunning

    Returns
    -------
    dict
        Dictionary containing all data cohorts used in modelling.
    """

    from sklearn.model_selection import train_test_split
    import pandas as pd

    random_state=42
    # Assign targets

    city_col, highway_col = target_cols


    # 1. Hold-out (Ford → OOD)

    train_val_df = df[df[make_col] != holdout_make].copy()
    ford_df      = df[df[make_col] == holdout_make].copy()


    # 2. Split NON-Ford → 60 / 40

    train_df_60, temp_df = train_test_split(train_val_df, test_size=0.40, random_state=random_state)


    # 3. Split 40 → 20 / 20

    val_df_20, test_df_20 = train_test_split(temp_df, test_size=0.50, random_state=random_state)


    # 4. Feature selection

    cols_to_drop = list(set(drop_cols + target_cols)) # i used set to avoid duplication of columns 

    X_train_60 = train_df_60.drop(columns=cols_to_drop)
    X_val_20   = val_df_20.drop(columns=cols_to_drop)
    X_test_20  = test_df_20.drop(columns=cols_to_drop)
    X_ford     = ford_df.drop(columns=cols_to_drop)


    # 5. Targets

    #y_train_60 = {"city": train_df_60[city_col], "highway": train_df_60[highway_col]}
    y_train_60_city = train_df_60[city_col]
    y_train_60_highway = train_df_60[highway_col]

    #y_val_20 = {"city": val_df_20[city_col], "highway": val_df_20[highway_col]}
    y_val_20_city = val_df_20[city_col]
    y_val_20_highway = val_df_20[highway_col]

    y_test_city_20    = test_df_20[city_col]
    y_test_highway_20 = test_df_20[highway_col]

    y_ford_city    = ford_df[city_col]
    y_ford_highway = ford_df[highway_col]


    # 6. Combine → 80% training

    X_train_80 = pd.concat([X_train_60, X_val_20], axis=0).reset_index(drop=True)

    #y_train_80 = {"city": pd.concat([y_train_60["city"], y_val_20["city"]], axis=0).reset_index(drop=True),
     #             "highway": pd.concat([y_train_60["highway"], y_val_20["highway"]], axis=0).reset_index(drop=True)}
    y_train_80_city = pd.concat([y_train_60_city, y_val_20_city], axis=0).reset_index(drop=True)
    y_train_80_highway = pd.concat([y_train_60_highway, y_val_20_highway], axis=0).reset_index(drop=True)

    # 7. ML ready to use outputs

    ml_cohorts = {"X_train_60": X_train_60,
                  "y_train_60_city": y_train_60_city,
                  "y_train_60_highway": y_train_60_highway,
                  "X_val_20": X_val_20,
                  "y_val_20_city": y_val_20_city,
                  "y_val_20_highway" : y_val_20_highway,
                  "X_test_20": X_test_20,
                  "y_test_city_20": y_test_city_20,
                  "y_test_highway_20": y_test_highway_20,
                  "X_train_80": X_train_80,
                  "y_train_80_city": y_train_80_city,
                  "y_train_80_highway": y_train_80_highway,
                  "X_ford": X_ford,
                  "y_ford_city": y_ford_city,
                  "y_ford_highway": y_ford_highway}

    return ml_cohorts
