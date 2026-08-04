"""
ml_utils.py
ML utilities for survey/assessment data:
 - K-means clustering of respondents based on item response patterns
 - Logistic regression to predict survey dropout/non-completion risk
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def cluster_respondents(item_df: pd.DataFrame, n_clusters: int = 3) -> dict:
    """
    Clusters respondents by their item response pattern using K-means.
    Returns cluster assignments and per-cluster mean profile.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(item_df)

    k = min(n_clusters, max(2, len(item_df) - 1))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    profile = item_df.copy()
    profile["cluster"] = labels
    cluster_means = profile.groupby("cluster").mean().round(2)

    return {
        "labels": labels.tolist(),
        "cluster_sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
        "cluster_profile": cluster_means.to_dict(orient="index"),
        "n_clusters": k,
    }


def predict_dropout_risk(df: pd.DataFrame, item_cols: list, target_col: str = "completed") -> dict:
    """
    Trains a logistic regression model predicting completion (1) vs dropout (0),
    using item responses (partial, if available) + response time as features.
    Returns model performance + per-respondent risk scores.
    """
    feature_cols = item_cols + (["response_time_sec"] if "response_time_sec" in df.columns else [])
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df[target_col]

    if y.nunique() < 2:
        return {"error": "Target column needs both classes (0 and 1) to train a model."}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr_s, ytr)

    pred = model.predict(Xte_s)
    proba = model.predict_proba(Xte_s)[:, 1]

    acc = accuracy_score(yte, pred)
    auc = roc_auc_score(yte, proba) if yte.nunique() > 1 else float("nan")

    # Risk score for every respondent (probability of completion; low = high dropout risk)
    all_scaled = scaler.transform(X)
    all_scores = model.predict_proba(all_scaled)[:, 1]

    return {
        "accuracy": round(float(acc), 3),
        "auc": round(float(auc), 3) if not np.isnan(auc) else None,
        "risk_scores": {int(rid): round(float(s), 3) for rid, s in zip(df.index, all_scores)},
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }
