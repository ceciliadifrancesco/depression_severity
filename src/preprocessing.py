#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df.copy()
    df.index = df.index.astype(str)
    return df

def extract_ids(df: pd.DataFrame) -> pd.DataFrame:
    df["participant_id"] = df.index.str.split("_").str[0].astype(int)
    df["month"] = df.index.str.split("_").str[1].astype(int)
    return df

def remove_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    patterns = ["score", "coeff", "intercept", "iqr", "range"]
    protected = ["phq9_score_start", "phq9_score_end"]
    cols_to_drop = [
        col for col in df.columns
        if any(p in col.lower() for p in patterns)
        and col not in protected
    ]
    return df.drop(columns=cols_to_drop)

def create_phq_variables(df: pd.DataFrame) -> pd.DataFrame:
    df["phq9_change"] = df["phq9_score_end"] - df["phq9_score_start"]
    df["phq_group"] = pd.cut(
        df["phq9_change"],
        bins=[-100, -2, 2, 100],
        labels=["Improved", "Stable", "Worsened"]
    )
    return df

def compute_age(df: pd.DataFrame, reference_year: int = 2019) -> pd.DataFrame:
    df["age"] = reference_year - df["birthyear"]
    return df

def add_migraine_indicator(df: pd.DataFrame) -> pd.DataFrame:
    df["has_migraine"] = (
        (df["comorbid_migraines"] == 1) |
        (df["num_migraine_days"] > 0)
    ).astype(int)
    return df

def preprocess(path: str) -> pd.DataFrame:
    df = load_data(path)
    df = extract_ids(df)
    df = remove_engineered_features(df)
    df = create_phq_variables(df)
    df = df[df["phq9_change"].notna()].copy()
    df = compute_age(df)

    # Add migraine indicator
    df = add_migraine_indicator(df)

    continuous_vars = [
        "steps_awake_mean", "sleep_asleep_weekday_mean", "sleep_asleep_weekend_mean",
        "sleep_in_bed_weekday_mean", "sleep_in_bed_weekend_mean",
        "sleep_ratio_asleep_in_bed_weekday_mean", "sleep_ratio_asleep_in_bed_weekend_mean",
        "sleep_main_start_hour_adj_median", "sleep_asleep_mean_recent",
        "sleep_in_bed_mean_recent", "sleep_ratio_asleep_in_bed_mean_recent",
        "steps_rolling_6_median_recent", "steps_rolling_6_max_recent",
        "educ", "height", "weight", "bmi", "pregnant", "birth", "insurance",
        "money", "money_assistance"
    ]

    median_imputer = SimpleImputer(strategy="median")
    df[continuous_vars] = median_imputer.fit_transform(df[continuous_vars])

    categorical_vars = ["sex"]
    freq_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_vars] = freq_imputer.fit_transform(df[categorical_vars])

    return df

if __name__ == "__main__":
    INPUT_PATH = r"C:\Users\cecil\Desktop\Digital health\project\depression_severity\data\raw\anon_processed_df_parquet"
    df_clean = preprocess(INPUT_PATH)
    OUTPUT_PATH = "data/processed/discover_cleaned.parquet"
    df_clean.to_parquet(OUTPUT_PATH)
    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")
