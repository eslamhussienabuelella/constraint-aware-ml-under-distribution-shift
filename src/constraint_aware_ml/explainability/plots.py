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

def get_lasso_feature_selection(model, X_reference):
    """
    Identify selected and eliminated features from a fitted Lasso pipeline.

    Parameters
    ----------
    model : sklearn Pipeline
        Fitted pipeline containing preprocessing and Lasso regression.
    X_reference : pd.DataFrame
        Original feature DataFrame used to fit the model.

    Returns
    -------
    dict
        Dictionary with selected and eliminated feature names.
    """

    import numpy as np

    # Get feature names AFTER preprocessing
    feature_names = model.named_steps['prep'].get_feature_names_out()

    # Get coefficients
    coefs = model.named_steps['reg'].coef_

    # Selected vs eliminated
    selected_features = [f for f, c in zip(feature_names, coefs) if not np.isclose(c, 0)]
    eliminated_features = [f for f, c in zip(feature_names, coefs) if np.isclose(c, 0)]

    return {"selected_features": selected_features,
            "eliminated_features": eliminated_features}

def generate_model_interpretability_charts(final_models, X_reference, top_n=15, shap_sample_size=300, save_path=None):

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import shap

    lr_model = final_models["LR"]
    lasso_model = final_models["Lasso"]
    rf_model = final_models["RF"]

    feature_names = lr_model.named_steps["prep"].get_feature_names_out()

    # -------------------------------
    # 1. Linear Regression
    # -------------------------------
    lr_coefs = np.ravel(lr_model.named_steps["reg"].coef_)

    lr_coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": lr_coefs
    }).assign(Abs=lambda df: df["Coefficient"].abs()) \
      .sort_values("Abs", ascending=False).head(top_n)

    plt.figure(figsize=(10, max(6, len(lr_coef_df) * 0.4)))
    plt.barh(lr_coef_df["Feature"], lr_coef_df["Coefficient"])
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Linear Regression Coefficients")
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "lr_coefficients.png"), dpi=300, bbox_inches="tight")

    plt.show()

    # -------------------------------
    # 2. Lasso
    # -------------------------------
    lasso_coefs = np.ravel(lasso_model.named_steps["reg"].coef_)

    lasso_coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": lasso_coefs
    }).assign(Abs=lambda df: df["Coefficient"].abs()) \
      .sort_values("Abs", ascending=False).head(top_n)

    plt.figure(figsize=(10, max(6, len(lasso_coef_df) * 0.4)))
    plt.barh(lasso_coef_df["Feature"], lasso_coef_df["Coefficient"])
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Lasso Coefficients")
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "lasso_coefficients.png"), dpi=300, bbox_inches="tight")

    plt.show()

    # -------------------------------
    # 3. LR vs Lasso
    # -------------------------------
    coef_compare_df = pd.DataFrame({
        "Feature": feature_names,
        "LR": lr_coefs,
        "Lasso": lasso_coefs
    })

    coef_compare_df["MaxAbsCoef"] = coef_compare_df[["LR", "Lasso"]].abs().max(axis=1)
    coef_compare_df = coef_compare_df.sort_values("MaxAbsCoef", ascending=False).head(top_n)

    y_pos = np.arange(len(coef_compare_df))

    plt.figure(figsize=(12, max(6, len(coef_compare_df) * 0.4)))
    plt.barh(y_pos - 0.2, coef_compare_df["LR"], height=0.4, label="LR")
    plt.barh(y_pos + 0.2, coef_compare_df["Lasso"], height=0.4, label="Lasso")
    plt.yticks(y_pos, coef_compare_df["Feature"])
    plt.gca().invert_yaxis()
    plt.title("LR vs Lasso Coefficients")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "lr_vs_lasso.png"), dpi=300, bbox_inches="tight")

    plt.show()

    # -------------------------------
    # 4. Random Forest Importance
    # -------------------------------
    rf_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.named_steps["reg"].feature_importances_
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, max(6, len(rf_importance_df) * 0.4)))
    plt.barh(rf_importance_df["Feature"], rf_importance_df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, "rf_importance.png"), dpi=300, bbox_inches="tight")

    plt.show()

    # -------------------------------
    # 5. SHAP
    # -------------------------------
    X_shap = X_reference.sample(min(shap_sample_size, len(X_reference)), random_state=42)

    X_transformed = rf_model.named_steps["prep"].transform(X_shap)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

    explainer = shap.TreeExplainer(rf_model.named_steps["reg"])
    shap_values = explainer.shap_values(X_transformed_df)

    # Beeswarm
    shap.summary_plot(shap_values, X_transformed_df, show=False)
    if save_path:
        plt.savefig(os.path.join(save_path, "rf_shap_beeswarm.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # Bar
    shap.summary_plot(shap_values, X_transformed_df, plot_type="bar", show=False)
    if save_path:
        plt.savefig(os.path.join(save_path, "rf_shap_bar.png"), dpi=300, bbox_inches="tight")
    plt.show()
