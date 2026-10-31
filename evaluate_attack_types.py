import os
import json
import pandas as pd
import numpy as np


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = "/app"

PARTITIONS_DIR = os.path.join(BASE_DIR, "dataset", "partitions")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

PREDICTIONS_FILE = os.path.join(
    ARTIFACTS_DIR,
    "federated_test_predictions.csv"
)

OUTPUT_CSV = os.path.join(
    ARTIFACTS_DIR,
    "attack_type_metrics.csv"
)

OUTPUT_HOME_CSV = os.path.join(
    ARTIFACTS_DIR,
    "attack_type_home_metrics.csv"
)

OUTPUT_JSON = os.path.join(
    ARTIFACTS_DIR,
    "attack_type_metrics.json"
)


# --------------------------------------------------
# LOAD FEDERATED PREDICTIONS
# --------------------------------------------------

print("\nLoading federated predictions...")

predictions = pd.read_csv(PREDICTIONS_FILE)

print("Prediction rows:", len(predictions))

required_columns = {
    "home_id",
    "device_id",
    "actual_label",
    "predicted_label",
    "attack_probability"
}

if not required_columns.issubset(predictions.columns):
    raise ValueError("Prediction CSV is missing required columns.")


# --------------------------------------------------
# MATCH PREDICTIONS WITH TEST METADATA
# --------------------------------------------------

all_records = []

for home_number in range(1, 6):

    home_id = f"home-{home_number}"

    test_file = os.path.join(
        PARTITIONS_DIR,
        f"{home_id}-test.csv"
    )

    if not os.path.exists(test_file):
        raise FileNotFoundError(test_file)

    test_data = pd.read_csv(test_file)

    home_predictions = predictions[
        predictions["home_id"] == home_id
    ].reset_index(drop=True)

    test_data = test_data.reset_index(drop=True)

    print(f"\nChecking {home_id}...")

    # Verify row counts
    if len(test_data) != len(home_predictions):
        raise ValueError(
            f"Row count mismatch for {home_id}: "
            f"{len(test_data)} test rows vs "
            f"{len(home_predictions)} predictions."
        )

    # Verify device IDs
    if not (test_data["device_id"].to_numpy() ==
            home_predictions["device_id"].to_numpy()).all():
        raise ValueError(f"Device ID mismatch in {home_id}")

    # Verify actual labels
    if not (test_data["label"].to_numpy() ==
            home_predictions["actual_label"].to_numpy()).all():
        raise ValueError(f"Label mismatch in {home_id}")

    if "attack_type" not in test_data.columns:
        raise ValueError(f"attack_type missing in {home_id}")

    print("Row alignment verified.")

    # Attach original metadata to predictions
    for index in range(len(test_data)):

        all_records.append({
            "home_id": home_id,
            "device_id": int(test_data.loc[index, "device_id"]),
            "attack_type": test_data.loc[index, "attack_type"],
            "actual_label": int(
                home_predictions.loc[index, "actual_label"]
            ),
            "predicted_label": int(
                home_predictions.loc[index, "predicted_label"]
            ),
            "attack_probability": float(
                home_predictions.loc[index, "attack_probability"]
            )
        })


# --------------------------------------------------
# CREATE COMBINED DATAFRAME
# --------------------------------------------------

results = pd.DataFrame(all_records)

print("\nTotal matched records:", len(results))

if len(results) != len(predictions):
    raise ValueError("Combined records do not match prediction count.")

print("All five home partitions successfully matched.")


# --------------------------------------------------
# ATTACK TYPE ANALYSIS
# --------------------------------------------------

attack_results = []

for attack_type, group in results.groupby("attack_type"):

    total = len(group)

    actual_attacks = group[group["actual_label"] == 1]

    true_positives = (
        (actual_attacks["predicted_label"] == 1).sum()
    )

    false_negatives = (
        (actual_attacks["predicted_label"] == 0).sum()
    )

    detection_rate = (
        true_positives / len(actual_attacks)
        if len(actual_attacks) > 0
        else None
    )

    false_negative_rate = (
        false_negatives / len(actual_attacks)
        if len(actual_attacks) > 0
        else None
    )

    attack_results.append({

        "attack_type": attack_type,

        "total_samples": total,

        "attack_samples": len(actual_attacks),

        "detected_attacks": int(true_positives),

        "missed_attacks": int(false_negatives),

        "detection_rate": detection_rate,

        "false_negative_rate": false_negative_rate

    })


attack_metrics = pd.DataFrame(attack_results)

attack_metrics = attack_metrics.sort_values(
    "detection_rate",
    ascending=True,
    na_position="last"
)


# --------------------------------------------------
# PER-HOME ATTACK TYPE ANALYSIS
# --------------------------------------------------

home_results = []

for (home_id, attack_type), group in results.groupby(
    ["home_id", "attack_type"]
):

    actual_attacks = group[group["actual_label"] == 1]

    if len(actual_attacks) == 0:
        continue

    detected = (
        actual_attacks["predicted_label"] == 1
    ).sum()

    missed = (
        actual_attacks["predicted_label"] == 0
    ).sum()

    home_results.append({

        "home_id": home_id,

        "attack_type": attack_type,

        "total_samples": len(group),

        "attack_samples": len(actual_attacks),

        "detected_attacks": int(detected),

        "missed_attacks": int(missed),

        "detection_rate": detected / len(actual_attacks)

    })


home_metrics = pd.DataFrame(home_results)


# --------------------------------------------------
# BENIGN TRAFFIC ANALYSIS
# --------------------------------------------------

benign = results[results["attack_type"] == "benign"]

benign_total = len(benign)

benign_false_alarms = (
    benign["predicted_label"] == 1
).sum()

benign_false_alarm_rate = (
    benign_false_alarms / benign_total
    if benign_total > 0
    else None
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

attack_metrics.to_csv(OUTPUT_CSV, index=False)

home_metrics.to_csv(OUTPUT_HOME_CSV, index=False)


summary = {

    "total_test_samples": len(results),

    "total_attack_samples": int(
        (results["actual_label"] == 1).sum()
    ),

    "total_benign_samples": int(
        (results["actual_label"] == 0).sum()
    ),

    "benign_false_alarms": int(benign_false_alarms),

    "benign_false_alarm_rate": benign_false_alarm_rate,

    "attack_type_metrics": attack_metrics.to_dict(
        orient="records"
    ),

    "home_attack_type_metrics": home_metrics.to_dict(
        orient="records"
    )

}


with open(OUTPUT_JSON, "w") as file:

    json.dump(summary, file, indent=4)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)

print("FEDERATED LEARNING - ATTACK TYPE EVALUATION")

print("=" * 65)

print(
    attack_metrics.to_string(
        index=False,
        formatters={
            "detection_rate": lambda x: f"{x:.4%}",
            "false_negative_rate": lambda x: f"{x:.4%}"
        }
    )
)

print("\nBENIGN TRAFFIC ANALYSIS")

print("Total benign samples:", benign_total)

print("False alarms:", benign_false_alarms)

print(
    "False alarm rate:",
    f"{benign_false_alarm_rate:.4%}"
    if benign_false_alarm_rate is not None
    else "N/A"
)

print("\nFILES GENERATED")

print(OUTPUT_CSV)

print(OUTPUT_HOME_CSV)

print(OUTPUT_JSON)

print("\nEvaluation completed successfully!")