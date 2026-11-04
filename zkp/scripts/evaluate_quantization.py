import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

SCALE = 10000

BASE_DIR = Path("..")

MODEL_PATH = BASE_DIR / "research-results/federated_global_model.json"
FEATURE_PATH = BASE_DIR / "research-results/feature_columns.json"
MEDIAN_PATH = BASE_DIR / "artifacts/client_preprocessing/home-1_medians.json"
DATASET_PATH = BASE_DIR / "dataset/partitions/home-1-test.csv"

with open(MODEL_PATH) as f:
    model = json.load(f)

with open(FEATURE_PATH) as f:
    features = json.load(f)

with open(MEDIAN_PATH) as f:
    preprocessing = json.load(f)

df = pd.read_csv(DATASET_PATH)

weights = np.array(model["weights"], dtype=np.float64)
bias = float(model["bias"])

medians = preprocessing["medians"]

# Select only the original model features
X_df = df[features].copy()

# Same preprocessing as home.py
X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_df = X_df.fillna(medians).fillna(0)

X = X_df.to_numpy(dtype=np.float64)

X = np.sign(X) * np.log1p(np.abs(X))

# Original floating-point predictions
original_logits = X @ weights + bias
original_predictions = (original_logits >= 0).astype(int)

# Fixed-point quantization
X_q = np.rint(X * SCALE).astype(np.int64)
W_q = np.rint(weights * SCALE).astype(np.int64)
B_q = int(np.rint(bias * SCALE))

# Quantized predictions
quantized_numerators = X_q @ W_q + B_q * SCALE
quantized_predictions = (quantized_numerators >= 0).astype(int)

# Dataset labels
y = df["label"].to_numpy(dtype=int)

agreement = np.mean(
    original_predictions == quantized_predictions
)

original_accuracy = accuracy_score(y, original_predictions)
quantized_accuracy = accuracy_score(y, quantized_predictions)

disagreements = np.where(
    original_predictions != quantized_predictions
)[0]

print("=" * 65)
print("FEDNATD - QUANTIZATION EVALUATION")
print("=" * 65)

print("Total test samples:", len(df))
print("Agreement:", agreement * 100, "%")
print("Disagreements:", len(disagreements))

print("\nModel accuracy against dataset labels")
print("-" * 65)

print("Floating-point accuracy:", original_accuracy * 100, "%")
print("Quantized accuracy:", quantized_accuracy * 100, "%")

print("\nDisagreement details")
print("-" * 65)

for idx in disagreements[:20]:
    print(
        f"Sample {idx}: "
        f"Original={original_predictions[idx]}, "
        f"Quantized={quantized_predictions[idx]}, "
        f"Label={y[idx]}, "
        f"Attack={df.iloc[idx]['attack_type']}"
    )

print("\nEvaluation completed.")