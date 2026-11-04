import pandas as pd
import numpy as np
from pathlib import Path

# Dataset location
DATASET_DIR = Path("../dataset/partitions")

files = sorted(DATASET_DIR.glob("*-train.csv"))

if not files:
    raise FileNotFoundError("No training CSV files found.")

print("=" * 65)
print("FEDNATD - FEATURE RANGE INSPECTION")
print("=" * 65)

for file in files:

    df = pd.read_csv(file)

    # Select model features
    X = df.drop(
        columns=["label", "attack_type", "device_id"],
        errors="ignore"
    )

    # Replace infinite values with NaN
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Client-local median imputation
    medians = X.median()

    X = X.fillna(medians).fillna(0)

    # Same transformation used in home.py
    X = np.sign(X) * np.log1p(np.abs(X))

    values = X.to_numpy(dtype=float)

    print(f"\nClient: {file.name}")
    print("-" * 65)

    print("Rows:", len(X))
    print("Features:", X.shape[1])

    print("Transformed minimum:", np.min(values))
    print("Transformed maximum:", np.max(values))

    print(
        "Maximum absolute value:",
        np.max(np.abs(values))
    )

    print(
        "Non-finite values:",
        np.sum(~np.isfinite(values))
    )

print("\nInspection completed.")