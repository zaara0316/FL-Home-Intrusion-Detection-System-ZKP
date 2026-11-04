import json
import numpy as np
from pathlib import Path

MODEL_PATH = Path("../research-results/federated_global_model.json")
DATASET_DIR = Path("../dataset/partitions")

SCALE = 10000

with open(MODEL_PATH, "r") as f:
    model = json.load(f)

weights = np.array(model["weights"], dtype=float)
bias = float(model["bias"])

quantized_weights = np.round(weights * SCALE).astype(np.int64)
quantized_bias = round(bias * SCALE)

max_abs_features = np.zeros(len(weights))

for file in sorted(DATASET_DIR.glob("*-train.csv")):
    import pandas as pd

    df = pd.read_csv(file)

    X = df.drop(
        columns=["label", "attack_type", "device_id"],
        errors="ignore"
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    medians = X.median()
    X = X.fillna(medians).fillna(0)

    X = np.sign(X) * np.log1p(np.abs(X))

    max_abs_features = np.maximum(
        max_abs_features,
        np.max(np.abs(X.to_numpy(dtype=float)), axis=0)
    )

quantized_feature_bounds = np.ceil(
    max_abs_features * SCALE
).astype(np.int64)

product_bounds = (
    np.abs(quantized_weights) * quantized_feature_bounds
)

max_dot_product = int(np.sum(product_bounds))

print("=" * 65)
print("FEDNATD - CIRCUIT ARITHMETIC BOUNDS")
print("=" * 65)

print("Number of features:", len(weights))
print("Scaling factor:", SCALE)

print("Maximum quantized feature:", np.max(quantized_feature_bounds))
print("Maximum quantized weight:", np.max(np.abs(quantized_weights)))

print("Maximum individual product bound:", np.max(product_bounds))
print("Maximum absolute dot-product bound:", max_dot_product)

print("Quantized bias:", quantized_bias)

print(
    "Maximum absolute numerator bound:",
    max_dot_product + abs(quantized_bias) * SCALE
)

print("\nInspection completed.")