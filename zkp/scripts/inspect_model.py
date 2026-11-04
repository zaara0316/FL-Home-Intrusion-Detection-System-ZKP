import json
import numpy as np
from pathlib import Path

MODEL_PATH = Path("../research-results/federated_global_model.json")

with open(MODEL_PATH, "r") as f:
    model = json.load(f)

weights = np.array(model["weights"], dtype=float)
bias = float(model["bias"])

print("=" * 65)
print("FEDNATD - GLOBAL MODEL INSPECTION")
print("=" * 65)

print("Number of weights:", len(weights))
print("Bias:", bias)

print("\nWeight statistics")
print("-" * 65)

print("Minimum weight:", np.min(weights))
print("Maximum weight:", np.max(weights))
print("Maximum absolute weight:", np.max(np.abs(weights)))

print("Mean absolute weight:", np.mean(np.abs(weights)))
print("Standard deviation:", np.std(weights))

print("\nQuantization preview (scale = 10000)")
print("-" * 65)

scale = 10000

quantized_weights = np.round(weights * scale).astype(np.int64)
quantized_bias = round(bias * scale)

print("Quantized weight minimum:", np.min(quantized_weights))
print("Quantized weight maximum:", np.max(quantized_weights))
print("Quantized bias:", quantized_bias)

print("\nMaximum absolute quantized weight:",
      np.max(np.abs(quantized_weights)))

print("\nInspection completed.")