import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


DATASET_PATH = "/dataset/processed/nbaiot_sample.csv"

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)

    metadata = [
        "device_id",
        "attack_type",
        "label"
    ]

    feature_columns = [
        col for col in df.columns
        if col not in metadata
    ]

    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    y = df["label"].astype(int)

    # Hold out device 5 entirely.
    train_mask = df["device_id"] != 5
    test_mask = df["device_id"] == 5

    X_train = X.loc[train_mask]
    X_test = X.loc[test_mask]

    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    print("\nTraining devices:")
    print(sorted(df.loc[train_mask, "device_id"].unique()))

    print("\nTesting device:")
    print(sorted(df.loc[test_mask, "device_id"].unique()))

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("\nTraining Logistic Regression...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    print("\n========== DEVICE-WISE RESULTS ==========")

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"FPR:       {fpr:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Benign", "Attack"],
        zero_division=0
    ))

    results = {
        "training_devices": [1, 2, 3, 4],
        "testing_device": 5,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "false_positive_rate": float(fpr),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    with open(
        ARTIFACT_DIR / "device_wise_metrics.json",
        "w"
    ) as file:

        json.dump(results, file, indent=4)

    print("\nResults saved to artifacts/device_wise_metrics.json")


if __name__ == "__main__":
    main()