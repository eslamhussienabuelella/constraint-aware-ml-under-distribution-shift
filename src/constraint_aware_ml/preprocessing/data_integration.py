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

def special_char_profiler(df):
    """
    Count special characters in all string columns.

    Parameters
    ----------
        df : pandas.DataFrame

    Returns
    -------
        dictionary where key ==> column name & value ==> special character counts
    """

    pattern = r"[^A-Za-z0-9,\s]"
    result = {}

    string_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in string_cols:
        counter = Counter()
        #print(counter)

        for val in df[col].dropna().astype(str):
            # count special characters
            chars = re.findall(pattern, val)
            counter.update(chars)
            # count double or multiple spaces
            double_spaces = re.findall(r" {2,}", val)
            if double_spaces:
                counter["  "] += len(double_spaces)
        if counter:
            #print(counter)
            #print(dict(counter))
            result[col] = dict(counter)
        else:
            result[col] = {"special charachters":0}
    return result

def veh_fc_specs_joiner(dfs_dict, input_year):
    """
    Fuzzy-match fuel consumption records with vehicle specification records for a given year using token-based Jaccard similarity staregy

    --------------------
    Parameters
    --------------------
    dfs_dict : Dictionary of DataFrames containing the vehicle specification & fuel consumption datasets.
    
    Must include:
        - "fc_df_11_23": fuel consumption data with tokenised `model_fuzzy_key`
        - yearly vehicle specification DataFrames keyed by year.

    input_year : int
        Model year used to filter fuel consumption records and select the corresponding
        vehicle specification dataset.
    
    --------------------
    Outputs (returns)
    --------------------
    
    1. fc_df_year_veh_spec_all:
            Final merged dataframe linking fuel consumption records with matched vehicle
            specifications for the specified year using fuzzy keys.
    2. Other outputs:
            Additional dataframes containing intermediate matching diagnostics.
    """
    fuzzy_best_match = {}

    import re
    
    weak_tokens = [re.compile(r"^v\d+$"),
                   re.compile(r"^\d+wd$"),
                   re.compile(r"^4x4$"),
                   re.compile(r"^awd$"),
                   re.compile(r"^fwd$"),
                   re.compile(r"^rwd$")]
    
    def remove_weak_tokens(token_set):
        """Remove weak tokens from a token set using regex matching."""
        strong_tokens = set()
    
        for token in token_set:
            token = str(token).lower().strip()
    
            if not any(pattern.fullmatch(token) for pattern in weak_tokens):
                strong_tokens.add(token)
        return strong_tokens
        
    vehicle_make_models= dfs_dict[input_year][["make_std","model_fuzzy_key"]].groupby("make_std")["model_fuzzy_key"].apply(list).to_dict()
    fc_df_year = dfs_dict["fc_df_11_23"][dfs_dict["fc_df_11_23"]["Model year"] == input_year].reset_index(drop=True)
    for i in range(len(fc_df_year)):
    #for i in range(len(dfs_dict["fc_df_11_23"][dfs_dict["fc_df_11_23"]["Model year"] == input_year])):
    
        fc_veh_make = str(fc_df_year.iloc[i, 1]).lower().strip()
        fc_veh_key  = fc_df_year.iloc[i, -1]
    
        # ensure set
    #    if not isinstance(fc_veh_key, set):
    #        fc_veh_key = set(str(fc_veh_key).lower().split())
    #
        if fc_veh_make in vehicle_make_models:
            tr_veh_spec_models = vehicle_make_models[fc_veh_make]
        else:
            print(f"no specs found for {fc_veh_make} in {input_year} vehicles")
            tr_veh_spec_models = set()
    
        # list to temporarily store all possible match scores (your variables)
        best_similarity_score = 0
        best_intersection_len = 0
        best_union_len = 0
        best_intersection = set()
        tie_counts = 0
    
    #    best_model_key = None  # not changing your list; just to avoid NameError
        best_model_key = set()  # not changing your list; just to avoid NameError
    
        for possible_model in tr_veh_spec_models:
            #if not isinstance(possible_model, set):
             #   possible_model = set(str(possible_model).lower().split())
    
            models_union = possible_model | fc_veh_key
            union_len = len(models_union)
            #if union_len == 0:
             #   continue
    
            models_intersection = possible_model & fc_veh_key
            intersection_len = len(models_intersection)
            similarity_ratio = intersection_len / union_len
    
            # new best
            if similarity_ratio > best_similarity_score:
                best_similarity_score = similarity_ratio
                best_intersection_len = intersection_len
                best_union_len = union_len
                best_intersection = models_intersection
                best_model_key = possible_model
                tie_counts = 0

########################################
            # tie -> choose best using tie-break rules
            elif similarity_ratio == best_similarity_score and best_similarity_score != 0:
                tie_counts += 1
            
                # tie-break 0 (stronger overlap): more non-weak tokens in intersection
                best_strong_tokens = remove_weak_tokens(best_intersection)
                curr_strong_tokens = remove_weak_tokens(models_intersection)
            
                best_strong = len(best_strong_tokens)
                curr_strong = len(curr_strong_tokens)
            
                if curr_strong > best_strong and len(curr_strong_tokens) != 0:
                    best_intersection_len = intersection_len
                    best_union_len = union_len
                    best_intersection = models_intersection
                    best_model_key = possible_model
            
                elif curr_strong == best_strong:
                    # tie-break 1: larger intersection
                    if intersection_len > best_intersection_len:
                        best_intersection_len = intersection_len
                        best_union_len = union_len
                        best_intersection = models_intersection
                        best_model_key = possible_model
            
                    # tie-break 2: smaller union
                    elif intersection_len == best_intersection_len and union_len < best_union_len:
                        best_union_len = union_len
                        best_intersection = models_intersection
                        best_model_key = possible_model
    
        # store best for this i (after checking all candidates)
        fuzzy_best_match[i] = [tie_counts, best_similarity_score, best_intersection, fc_veh_key, best_model_key]
    model_matching_df = pd.DataFrame(fuzzy_best_match).transpose()
    header = ["ties", "similarity ratio", "join intersection", "fc_join_key", "veh_spec_join_key"]
    model_matching_df.columns = header
    fc_df_year = dfs_dict["fc_df_11_23"][dfs_dict["fc_df_11_23"]["Model year"] == input_year].reset_index(drop=True)
    veh_spec_year_df = dfs_dict[input_year].copy()
    fc_df_year_all_keys = pd.concat([fc_df_year, model_matching_df], axis=1)

    # make join keys hashable for merge & value counts
        # ifinstance in ==> lambda key: frozenset(key) if isinstance(key, set) else key) to handle none set keys if existed
    fc_df_year_all_keys["veh_spec_join_key_hash"] = fc_df_year_all_keys["veh_spec_join_key"].apply(
        lambda key: frozenset(key) if isinstance(key, set) else key)
    veh_spec_year_df["model_fuzzy_key_hash"] = veh_spec_year_df["model_fuzzy_key"].apply(
        lambda key: frozenset(key) if isinstance(key, set) else key)

    fc_df_year_veh_spec_all = pd.merge(
        fc_df_year_all_keys,
        veh_spec_year_df,
        how="left",
        left_on="veh_spec_join_key_hash",
        right_on="model_fuzzy_key_hash"
    )

    #fc_df_year_veh_spec_all = pd.merge(fc_df_year, veh_spec_year_df, how='left', on='model_fuzzy_key')
    return(fc_df_year_veh_spec_all)
