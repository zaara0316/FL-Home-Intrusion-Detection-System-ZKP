import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LABEL_CANDIDATES = ("label", "Label", "class", "Class", "target", "Target", "attack")

def load_csv_dataset(path, n_features):
    # Optional convenience loader. Validate label mapping and feature schema
    # against the chosen dataset before using this for research.
    if not path or not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"No label column found. Expected one of {LABEL_CANDIDATES}")
    raw = df[label_col]
    if raw.dtype == object:
        normalized = raw.astype(str).str.strip().str.lower()
        y = (~normalized.isin({"benign", "normal", "0", "false"})).astype(int).to_numpy()
    else:
        unique = set(pd.Series(raw).dropna().unique().tolist())
        if unique.issubset({0, 1}):
            y = raw.astype(int).to_numpy()
        else:
            majority = pd.Series(raw).value_counts().idxmax()
            y = (raw != majority).astype(int).to_numpy()
    features = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
    if features.shape[1] < n_features:
        raise ValueError(f"Only {features.shape[1]} numeric features; need {n_features}")
    X = features.iloc[:, :n_features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)
    return X, y
