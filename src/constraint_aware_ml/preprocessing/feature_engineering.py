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

def add_physics_features(df_dict):
    #df = df.copy()
    for key, df in df_dict.items():
        if key[0] == "X":
            # Power-to-Weight Proxy
            df["power_to_weight_proxy"] = (df["engine_size_l"] * df["cylinders"]) / df["curb_weight"]
            
            # Mass–Aerodynamic Interaction
            df["mass_aero_interaction"] = (df["curb_weight"] * df["overall_width"] * df["overall_height"])
        
    return df_dict
