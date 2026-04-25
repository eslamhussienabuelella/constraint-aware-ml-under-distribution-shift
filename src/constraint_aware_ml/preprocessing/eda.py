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

def plot_combined_vehicle_specs(df):
    # 1. Define the groups in a dictionary to maintain order and titles
    groups = {
        "Fuel Consumption (L/100km)": ["city_l_per_100_km", "highway_l_per_100_km"],
        "Engine Specifications": ["engine_size_l", "cylinders"],
        "Overall Vehicle Geometry": ["overall_length", "overall_width", "overall_height", "wheelbase"],
        "Body Dimensions": ["front_end_length", "rear_end_length", "side_glass_height", 
                            "body_side_height", "roof_width", "front_overhang", 
                            "rear_overhang", "track_width_front", "track_width_rear", 
                            "weight_distribution_front", "weight_distribution_rear"],
        "Mass Variables": ["curb_weight"]
    }

    # 2. Create a figure with a vertical stack of subplots
    # We use len(groups) to dynamically set the number of rows
    fig, axes = plt.subplots(nrows=len(groups), ncols=1, figsize=(12, 5 * len(groups)))
    
    # 3. Iterate through the dictionary and the axes simultaneously
    for ax, (title, cols) in zip(axes, groups.items()):
        # Plot horizontal boxplot
        sns.boxplot(data=df[cols], orient="h", ax=ax, palette="viridis", fliersize=4)
        
        # Consistent styling for each subplot
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel("Value / Unit", fontsize=10)
        ax.tick_params(axis='y', labelsize=10)

    # 4. Global layout adjustments
    plt.suptitle("Vehicle Specification Outlier Analysis", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_grouped_boxplot(df, value_col, group_col, title):
    # Set a larger figure size for better spacing
    plt.figure(figsize=(12, 10)) 
    
    # Sort categories by median for smooth visual flow
    order = df.groupby(group_col)[value_col].median().sort_values().index
    #viridis
    ax = sns.boxplot(data=df, x=value_col, y=group_col, hue = group_col, hue_order = order, order=order, palette="viridis", fliersize=6)
    
    # Remove the redundant legend created by 'hue'
    #if ax.get_legend():
   #     ax.get_legend().remove()

    # --- Annotation ---
    # Shared arrow style for a clean, consistent look
    #arrow_style = dict(arrowstyle='->', connectionstyle='arc3,rad=.3', color='black', lw=1.2)
    arrow_style=dict(arrowstyle="->", color="black")
    
    # 1. Label the Low-End Error (Kia Carnival @ 109kg)
    if "Minivan" in order:
        idx = list(order).index("Minivan")
        plt.annotate('Data Error: Kia Carnival (109kg)', 
                     xy=(109, idx), xytext=(50, idx - 0.7),
                     arrowprops=arrow_style,
                     fontsize=11, color='darkred', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9))

    # 2. Label the High-End Error (VW Atlas @ 4288kg/lbs)
    if "Sport utility vehicle: Small" in order:
        idx = list(order).index("Sport utility vehicle: Small")
        plt.annotate('Unit Error: VW Atlas (4288 lbs vs kg)', 
                     xy=(4288, idx), xytext=(2800, idx + 0.8),
                     arrowprops=arrow_style,
                     fontsize=11, color='darkred', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9))

    # 3. Label Representative Outliers (Smart Fortwo @ ~750kg)
    if "Two-seater" in order:
        idx = list(order).index("Two-seater")
        plt.annotate('Representative: Smart Fortwo', 
                     xy=(750, idx), xytext=(50, idx - 1.5),
                     arrowprops=arrow_style,
                     fontsize=11, color='darkgreen', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.9))
    
    # figure titles
    plt.title(title, fontsize=16, pad=25, fontweight='bold')
    plt.xlabel("Curb Weight (kg)", fontsize=13)
    plt.ylabel("Vehicle Class", fontsize=13)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Limit x-axis to keep the chart centred
    #plt.xlim(-100, 5200) 
    plt.tight_layout()
    plt.show()

def plot_categorical_distribution(df, cols, figsize=(10, 6)):
    """
    Plots a sorted horizontal barchart for each categorical column in the list with numeric labels.
    """
    for col in cols:
        if col in df.columns:
            plt.figure(figsize=figsize)
            
            # Get the order by frequency (descending)
            order = df[col].value_counts().index
            
            # Create the plot and assign to 'ax'
            ax = sns.countplot(data=df, y=col, hue=col, hue_order=order, order=order, palette="viridis", legend=False)
            
            # Add numeric labels to each bar
            for container in ax.containers:
                ax.bar_label(container, padding=3)
            
            plt.title(f"Distribution of {col.replace('_', ' ').title()} (Sorted)")
            plt.xlabel("Count")
            plt.ylabel(col.replace('_', ' ').title())
            
            # Adjust x-axis limit slightly so labels don't get cut off
            ax.set_xlim(right=ax.get_xlim()[1] * 1.1)
            
            plt.tight_layout()
            plt.show()
        else:
            print(f"Column '{col}' not found in DataFrame.")
