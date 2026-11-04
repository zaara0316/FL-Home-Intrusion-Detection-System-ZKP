import json
import copy
from pathlib import Path

INPUT_PATH = Path("inputs/inference_input.json")
OUTPUT_DIR = Path("inputs")

with open(INPUT_PATH) as f:
    original = json.load(f)

# Test 1: Tamper with prediction
prediction_test = copy.deepcopy(original)

prediction_test["prediction"] = str(
    1 - int(prediction_test["prediction"])
)

with open(OUTPUT_DIR / "tampered_prediction.json", "w") as f:
    json.dump(prediction_test, f, indent=2)

# Test 2: Tamper with model weight
weight_test = copy.deepcopy(original)

weight_test["weightMag"][0] = str(
    int(weight_test["weightMag"][0]) + 1
)

with open(OUTPUT_DIR / "tampered_weight.json", "w") as f:
    json.dump(weight_test, f, indent=2)

# Test 3: Tamper with one feature
feature_test = copy.deepcopy(original)

feature_test["featureMag"][0] = str(
    int(feature_test["featureMag"][0]) + 1
)

with open(OUTPUT_DIR / "tampered_feature.json", "w") as f:
    json.dump(feature_test, f, indent=2)

print("Tampering test inputs generated.")

for filename in [
    "tampered_prediction.json",
    "tampered_weight.json",
    "tampered_feature.json"
]:
    print("Created:", filename)