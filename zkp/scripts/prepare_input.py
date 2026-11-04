import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SCALE = 10000
SAMPLE_INDEX = int(sys.argv[1]) if len(sys.argv) > 1 else 0

BASE_DIR = Path("..")

MODEL_PATH = BASE_DIR / "research-results/federated_global_model.json"
FEATURE_PATH = BASE_DIR / "research-results/feature_columns.json"
MEDIAN_PATH = BASE_DIR / "artifacts/client_preprocessing/home-1_medians.json"
DATASET_PATH = BASE_DIR / "dataset/partitions/home-1-test.csv"

OUTPUT_DIR = Path("inputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "inference_input.json"

# --------------------------------------------------
# LOAD FILES
# --------------------------------------------------

print("=" * 65)
print("FEDNATD - ZKP INPUT PREPARATION")
print("=" * 65)

with open(MODEL_PATH) as f:
    model = json.load(f)

with open(FEATURE_PATH) as f:
    feature_columns = json.load(f)

with open(MEDIAN_PATH) as f:
    preprocessing = json.load(f)

df = pd.read_csv(DATASET_PATH)

weights = np.asarray(model["weights"], dtype=np.float64)
bias = float(model["bias"])

medians = preprocessing["medians"]

# --------------------------------------------------
# VALIDATE FEATURE SCHEMA
# --------------------------------------------------

assert len(feature_columns) == 115
assert len(weights) == 115

assert feature_columns == preprocessing["feature_columns"]

assert all(column in df.columns for column in feature_columns)

print("Feature schema validated.")
print("Number of features:", len(feature_columns))

# --------------------------------------------------
# SELECT TEST SAMPLE
# --------------------------------------------------

if SAMPLE_INDEX >= len(df):
    raise IndexError("Sample index exceeds dataset size.")

sample = df.iloc[SAMPLE_INDEX]

original_label = int(sample["label"])
attack_type = str(sample["attack_type"])

print("\nSelected sample")
print("-" * 65)
print("Sample index:", SAMPLE_INDEX)
print("Original label:", original_label)
print("Attack type:", attack_type)

# --------------------------------------------------
# APPLY CLIENT PREPROCESSING
# --------------------------------------------------

X_df = sample[feature_columns].to_frame().T.copy()

X_df = X_df.replace([np.inf, -np.inf], np.nan)

X_df = X_df.fillna(medians).fillna(0)

X = X_df.to_numpy(dtype=np.float64)

# Same transformation used in home.py
X = np.sign(X) * np.log1p(np.abs(X))

assert np.all(np.isfinite(X))

# --------------------------------------------------
# ORIGINAL FLOATING-POINT INFERENCE
# --------------------------------------------------

original_logit = float(X[0] @ weights + bias)

original_prediction = int(original_logit >= 0)

# --------------------------------------------------
# FIXED-POINT QUANTIZATION
# --------------------------------------------------

feature_q = np.rint(X[0] * SCALE).astype(np.int64)

weight_q = np.rint(weights * SCALE).astype(np.int64)

bias_q = int(np.rint(bias * SCALE))

# --------------------------------------------------
# QUANTIZED INFERENCE
# --------------------------------------------------

quantized_numerator = int(
    np.dot(feature_q, weight_q) + bias_q * SCALE
)

quantized_prediction = int(quantized_numerator >= 0)

quantized_logit = quantized_numerator / (SCALE * SCALE)

# --------------------------------------------------
# CHECK CIRCUIT INPUT BOUNDS
# --------------------------------------------------

if np.max(np.abs(feature_q)) >= 2**19:
    raise ValueError("Feature magnitude exceeds 19-bit circuit range.")

if np.max(np.abs(weight_q)) >= 2**10:
    raise ValueError("Weight magnitude exceeds 10-bit circuit range.")

if abs(bias_q) >= 2**7:
    raise ValueError("Bias magnitude exceeds 7-bit circuit range.")

# --------------------------------------------------
# SIGN / MAGNITUDE ENCODING
# --------------------------------------------------

feature_mag = np.abs(feature_q).astype(int).tolist()

feature_sign = (feature_q < 0).astype(int).tolist()

weight_mag = np.abs(weight_q).astype(int).tolist()

weight_sign = (weight_q < 0).astype(int).tolist()

bias_mag = abs(bias_q)

bias_sign = int(bias_q < 0)

# --------------------------------------------------
# GENERATE CIRCUIT INPUT
# --------------------------------------------------

circuit_input = {
    "modelCommitment": "0",
    "prediction": str(quantized_prediction),

    "featureMag": [str(x) for x in feature_mag],
    "featureSign": [str(x) for x in feature_sign],

    "weightMag": [str(x) for x in weight_mag],
    "weightSign": [str(x) for x in weight_sign],

    "biasMag": str(bias_mag),
    "biasSign": str(bias_sign)
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(circuit_input, f, indent=2)

# --------------------------------------------------
# REPORT
# --------------------------------------------------

print("\nInference results")
print("-" * 65)

print("Original floating-point logit:", original_logit)
print("Quantized logit:", quantized_logit)

print("Original prediction:", original_prediction)
print("Quantized prediction:", quantized_prediction)

print("Original dataset label:", original_label)

print(
    "Floating-point and quantized predictions match:",
    original_prediction == quantized_prediction
)

print("\nQuantization error:", abs(original_logit - quantized_logit))

print("\nCircuit input statistics")
print("-" * 65)

print("Maximum feature magnitude:", max(feature_mag))
print("Maximum weight magnitude:", max(weight_mag))
print("Bias magnitude:", bias_mag)

print("\nSaved input:", OUTPUT_PATH)
print("\nInput preparation completed.")

if original_prediction != quantized_prediction:
    print(
        "\nWARNING: Quantized prediction differs from "
        "floating-point prediction."
    )
