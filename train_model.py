import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


DATASET_PATH = "/dataset/processed/nbaiot_sample.csv"

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_dataset():

    print("\nLoading processed dataset...")

    df = pd.read_csv(DATASET_PATH)

    metadata_columns = [
        "device_id",
        "attack_type",
        "label"
    ]

    feature_columns = [
        col for col in df.columns
        if col not in metadata_columns
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

    print("\nDataset information:")
    print("Samples:", len(df))
    print("Features:", len(feature_columns))

    print("\nLabel distribution:")
    print(y.value_counts())

    print("\nFeature columns:", feature_columns)

    return X, y, feature_columns


def evaluate_model(name, model, X_test, y_test):

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

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

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

    return {
        "model": name,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "false_positive_rate": round(float(fpr), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }


def main():

    X, y, feature_columns = load_dataset()

    if len(y.unique()) != 2:
        raise ValueError(
            "Expected two classes: benign and attack."
        )

    # Stratified split preserves the class proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    models = {

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])

    }

    results = []

    trained_models = {}

    for name, model in models.items():

        print("\nTraining:", name)

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            name,
            model,
            X_test,
            y_test
        )

        results.append(metrics)

        trained_models[name] = model

    # Select the best model based on F1-score.
    best_result = max(
        results,
        key=lambda item: item["f1_score"]
    )

    best_name = best_result["model"]

    best_model = trained_models[best_name]

    model_path = ARTIFACT_DIR / "baseline_model.joblib"

    joblib.dump(best_model, model_path)

    metrics_path = ARTIFACT_DIR / "baseline_metrics.json"

    with open(metrics_path, "w") as file:

        json.dump(results, file, indent=4)

    # Save the feature schema for subsequent experiments.
    schema_path = ARTIFACT_DIR / "feature_columns.json"

    with open(schema_path, "w") as file:

        json.dump(feature_columns, file, indent=4)

    print("\n" + "=" * 55)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 55)

    print("Best model:", best_name)
    print("Best F1-score:", best_result["f1_score"])

    print("\nSaved artifacts:")
    print(model_path)
    print(metrics_path)
    print(schema_path)


if __name__ == "__main__":
    main()