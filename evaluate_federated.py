import os
import json

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# CONFIGURATION
# ==========================================================

ARTIFACT_DIR = "/app/artifacts"

PARTITION_DIR = "/app/dataset/partitions"

MODEL_PATH = os.path.join(
    ARTIFACT_DIR,
    "federated_global_model.json"
)

FEATURE_PATH = os.path.join(
    ARTIFACT_DIR,
    "feature_columns.json"
)

OUTPUT_JSON = os.path.join(
    ARTIFACT_DIR,
    "federated_test_metrics.json"
)

OUTPUT_CSV = os.path.join(
    ARTIFACT_DIR,
    "federated_test_predictions.csv"
)


# ==========================================================
# LOAD GLOBAL MODEL
# ==========================================================

print("\nLoading federated global model...")

with open(MODEL_PATH, "r") as file:
    model = json.load(file)

weights = np.asarray(
    model["weights"],
    dtype=float
)

bias = float(model["bias"])

print("Global round:", model["round"])
print("Number of model weights:", len(weights))


# ==========================================================
# LOAD FEATURE SCHEMA
# ==========================================================

with open(FEATURE_PATH, "r") as file:
    feature_columns = json.load(file)


if len(feature_columns) != len(weights):
    raise ValueError(
        "Feature count does not match model weight count."
    )


# ==========================================================
# FEATURE TRANSFORMATION
# ==========================================================

def transform_features(X):

    return np.sign(X) * np.log1p(np.abs(X))


def sigmoid(z):

    z = np.clip(z, -35, 35)

    return 1.0 / (1.0 + np.exp(-z))


# ==========================================================
# METRICS
# ==========================================================

def calculate_metrics(y_true, predictions, probabilities):

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {

        "samples": int(len(y_true)),

        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),

        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities
            )
        ),

        "false_positive_rate": float(fpr),

        "confusion_matrix": {

            "true_negative": int(tn),

            "false_positive": int(fp),

            "false_negative": int(fn),

            "true_positive": int(tp)
        },

        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=["benign", "attack"],
            output_dict=True,
            zero_division=0
        )
    }


# ==========================================================
# EVALUATE EACH HOME
# ==========================================================

print("\n" + "=" * 60)
print("FEDERATED GLOBAL MODEL EVALUATION")
print("=" * 60)

all_y = []
all_predictions = []
all_probabilities = []

home_results = []
prediction_records = []

for device_id in range(1, 6):

    home_id = f"home-{device_id}"

    print("\nEvaluating:", home_id)

    test_path = os.path.join(
        PARTITION_DIR,
        f"{home_id}-test.csv"
    )

    median_path = os.path.join(
        ARTIFACT_DIR,
        "client_preprocessing",
        f"{home_id}_medians.json"
    )

    test_df = pd.read_csv(test_path)

    with open(median_path, "r") as file:
        preprocessing = json.load(file)

    # Verify device assignment.
    device_ids = test_df["device_id"].astype(str).unique()

    if len(device_ids) != 1 or device_ids[0] != str(device_id):
        raise ValueError(
            f"Incorrect test partition for {home_id}"
        )

    # Load the training-derived imputation values.
    medians = pd.Series(
        preprocessing["medians"]
    )

    X_df = test_df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_df = X_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Apply training-derived medians.
    X_df = X_df.fillna(medians).fillna(0)

    X = transform_features(
        X_df.to_numpy(dtype=float)
    )

    y = test_df["label"].to_numpy(dtype=int)

    # Global model inference.
    probabilities = sigmoid(
        X @ weights + bias
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities
    )

    metrics["home_id"] = home_id

    home_results.append(metrics)

    all_y.extend(y.tolist())

    all_predictions.extend(
        predictions.tolist()
    )

    all_probabilities.extend(
        probabilities.tolist()
    )

    # Preserve individual predictions.
    for index in range(len(y)):

        prediction_records.append({

            "home_id": home_id,

            "device_id": device_id,

            "actual_label": int(y[index]),

            "predicted_label": int(predictions[index]),

            "attack_probability": float(
                probabilities[index]
            )
        })

    print(
        f"Samples: {metrics['samples']}"
    )

    print(
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {metrics['precision']:.4f}"
    )

    print(
        f"Recall: {metrics['recall']:.4f}"
    )

    print(
        f"F1-score: {metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC: {metrics['roc_auc']:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{metrics['false_positive_rate']:.4f}"
    )


# ==========================================================
# OVERALL EVALUATION
# ==========================================================

overall_metrics = calculate_metrics(
    np.asarray(all_y),
    np.asarray(all_predictions),
    np.asarray(all_probabilities)
)

print("\n" + "=" * 60)
print("OVERALL FEDERATED MODEL PERFORMANCE")
print("=" * 60)

for metric in [
    "samples",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "false_positive_rate"
]:

    print(
        f"{metric}: {overall_metrics[metric]}"
    )


# ==========================================================
# SAVE RESULTS
# ==========================================================

results = {

    "experiment": "FedNatD Federated Evaluation",

    "model": "Federated Logistic Regression",

    "global_round": model["round"],

    "threshold": 0.5,

    "total_test_samples": len(all_y),

    "per_home_results": home_results,

    "overall_results": overall_metrics

}

with open(OUTPUT_JSON, "w") as file:

    json.dump(
        results,
        file,
        indent=4
    )


pd.DataFrame(
    prediction_records
).to_csv(
    OUTPUT_CSV,
    index=False
)


print("\nEvaluation completed.")

print("Saved metrics:", OUTPUT_JSON)

print("Saved predictions:", OUTPUT_CSV)