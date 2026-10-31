import os
import json

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


BASE_DIR = "/app"

PARTITION_DIR = os.path.join(
    BASE_DIR, "dataset", "partitions"
)

ARTIFACT_DIR = os.path.join(
    BASE_DIR, "artifacts"
)

FEATURE_PATH = os.path.join(
    ARTIFACT_DIR, "feature_columns.json"
)


with open(FEATURE_PATH) as f:
    feature_columns = json.load(f)


train_frames = []
test_frames = []

for device_id in range(1, 6):

    home_id = f"home-{device_id}"

    train_frames.append(
        pd.read_csv(
            os.path.join(
                PARTITION_DIR,
                f"{home_id}-train.csv"
            )
        )
    )

    test_frames.append(
        pd.read_csv(
            os.path.join(
                PARTITION_DIR,
                f"{home_id}-test.csv"
            )
        )
    )


train_df = pd.concat(
    train_frames,
    ignore_index=True
)

test_df = pd.concat(
    test_frames,
    ignore_index=True
)


X_train = train_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = test_df[feature_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

y_train = train_df["label"].to_numpy(dtype=int)

y_test = test_df["label"].to_numpy(dtype=int)


# Same signed logarithmic transformation.
X_train = np.sign(X_train) * np.log1p(np.abs(X_train))

X_test = np.sign(X_test) * np.log1p(np.abs(X_test))


# Fit preprocessing only on training data.
model = Pipeline([

    ("imputer", SimpleImputer(strategy="median")),

    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))

])


print("Training controlled centralized model...")

model.fit(X_train, y_train)


probabilities = model.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0


results = {

    "model": "Controlled Centralized Logistic Regression",

    "training_samples": len(y_train),

    "testing_samples": len(y_test),

    "accuracy": float(
        accuracy_score(y_test, predictions)
    ),

    "precision": float(
        precision_score(y_test, predictions)
    ),

    "recall": float(
        recall_score(y_test, predictions)
    ),

    "f1_score": float(
        f1_score(y_test, predictions)
    ),

    "roc_auc": float(
        roc_auc_score(y_test, probabilities)
    ),

    "false_positive_rate": float(fpr),

    "confusion_matrix": {

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp)

    }

}


output_path = os.path.join(
    ARTIFACT_DIR,
    "controlled_centralized_metrics.json"
)

with open(output_path, "w") as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\nCONTROLLED CENTRALIZED RESULTS")

for key, value in results.items():

    print(f"{key}: {value}")


print("\nSaved:", output_path)