"""
stats_utils.py
Psychometric / statistical utilities for survey and assessment data:
 - Cronbach's alpha (internal consistency reliability)
 - Item difficulty (mean item score, normalized)
 - Item-total correlation (discrimination index)
 - Basic descriptive stats
"""

import numpy as np
import pandas as pd


def cronbach_alpha(item_df: pd.DataFrame) -> float:
    """
    Computes Cronbach's alpha for a DataFrame where each column is one survey/test item.
    """
    item_df = item_df.dropna()
    k = item_df.shape[1]
    if k < 2:
        return float("nan")
    item_variances = item_df.var(axis=0, ddof=1)
    total_scores = item_df.sum(axis=1)
    total_variance = total_scores.var(ddof=1)
    if total_variance == 0:
        return float("nan")
    alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    return float(alpha)


def item_difficulty(item_df: pd.DataFrame, scale_max: float) -> pd.Series:
    """
    Item difficulty = mean item score / max possible score.
    For Likert data, values near 0 or 1 indicate floor/ceiling effects (very hard/easy items).
    """
    return (item_df.mean(axis=0) / scale_max).round(3)


def item_discrimination(item_df: pd.DataFrame) -> pd.Series:
    """
    Item-total correlation: correlation of each item with the sum of all OTHER items.
    Values below ~0.2 suggest the item isn't discriminating well and may need revision.
    """
    results = {}
    for col in item_df.columns:
        rest_total = item_df.drop(columns=[col]).sum(axis=1)
        results[col] = item_df[col].corr(rest_total)
    return pd.Series(results).round(3)


def descriptive_summary(item_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "mean": item_df.mean(),
        "std": item_df.std(),
        "min": item_df.min(),
        "max": item_df.max(),
    }).round(3)
    return summary


def full_report(item_df: pd.DataFrame, scale_max: float = 5) -> dict:
    return {
        "cronbach_alpha": round(cronbach_alpha(item_df), 3),
        "item_difficulty": item_difficulty(item_df, scale_max).to_dict(),
        "item_discrimination": item_discrimination(item_df).to_dict(),
        "descriptives": descriptive_summary(item_df).to_dict(orient="index"),
        "n_respondents": int(item_df.shape[0]),
        "n_items": int(item_df.shape[1]),
    }
