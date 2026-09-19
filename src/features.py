from __future__ import annotations
import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature engineering shared by training and inference."""
    out = df.copy()
    out = out.drop(columns=["customer_id"], errors="ignore")
    if {"balance", "products_number"}.issubset(out.columns):
        out["balance_per_product"] = (
            out["balance"] / out["products_number"].replace(0, np.nan)
        ).fillna(0.0)
    if {"estimated_salary", "balance"}.issubset(out.columns):
        out["salary_balance_ratio"] = (
            out["estimated_salary"]
            / out["balance"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "age" in out.columns:
        out["age_group"] = pd.cut(
            out["age"], [0, 25, 35, 45, 55, 65, 100],
            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
            include_lowest=True,
        ).astype("object")
    if "tenure" in out.columns:
        out["tenure_bucket"] = pd.cut(
            out["tenure"], [-1, 0, 2, 5, 10, np.inf],
            labels=["0", "1-2", "3-5", "6-10", "10+"],
        ).astype("object")
    if "balance" in out.columns:
        # Fixed threshold prevents leakage from train/test quantiles.
        out["high_balance"] = (out["balance"] >= 100000.0).astype(int)
    return out
