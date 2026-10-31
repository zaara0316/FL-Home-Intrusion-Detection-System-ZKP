import os
import time
import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

HOME_ID = os.getenv("HOME_ID", "home-1")

DEVICE_ID = int(os.getenv("DEVICE_ID", "1"))

SERVER_URL = os.getenv(
    "FL_SERVER_URL",
    "http://localhost:8000"
).rstrip("/")

DATASET_PATH = os.getenv(
    "DATASET_PATH",
    "/data/train.csv"
)

FEATURE_SCHEMA_PATH = os.getenv(
    "FEATURE_SCHEMA_PATH",
    "/app/artifacts/feature_columns.json"
)

TRAIN_ROUNDS = int(os.getenv("TRAIN_ROUNDS", "5"))

LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "3"))

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))

L2 = float(os.getenv("L2", "0.0001"))


# ============================================================
# LOGGING
# ============================================================

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/{HOME_ID}.log")
    ]
)

log = logging.getLogger(HOME_ID)


# ============================================================
# SERVER CONNECTION
# ============================================================

def wait_for_server():

    for attempt in range(60):

        try:

            response = requests.get(
                f"{SERVER_URL}/health",
                timeout=3
            )

            response.raise_for_status()

            log.info("Server connected.")

            return

        except requests.RequestException:

            log.info(
                "Waiting for server (%d/60)...",
                attempt + 1
            )

            time.sleep(2)

    raise RuntimeError("FL server unavailable")


# ============================================================
# FEATURE TRANSFORMATION
# ============================================================

def transform_features(X):

    return np.sign(X) * np.log1p(np.abs(X))


# ============================================================
# LOAD CLIENT TRAINING DATA
# ============================================================

def load_local_data():

    log.info("Loading training partition: %s", DATASET_PATH)

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(DATASET_PATH)

    with open(FEATURE_SCHEMA_PATH, "r") as file:

        feature_columns = json.load(file)

    df = pd.read_csv(DATASET_PATH)

    if "label" not in df.columns:

        raise ValueError("Label column missing.")

    if "device_id" not in df.columns:

        raise ValueError("Device ID column missing.")

    # Confirm the partition belongs to this client.
    device_ids = df["device_id"].astype(str).unique()

    if len(device_ids) != 1 or device_ids[0] != str(DEVICE_ID):

        raise ValueError(
            f"{HOME_ID} received an incorrect device partition: "
            f"{device_ids}"
        )

    X_df = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_df = X_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fit imputation values using training data only.
    medians = X_df.median()

    X_df = X_df.fillna(medians).fillna(0)

    # Save local preprocessing parameters for evaluation.
    preprocessing_dir = Path(
        "/app/artifacts/client_preprocessing"
    )

    preprocessing_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    preprocessing_file = (
        preprocessing_dir / f"{HOME_ID}_medians.json"
    )

    with open(preprocessing_file, "w") as file:

        json.dump(
            {
                "home_id": HOME_ID,
                "device_id": DEVICE_ID,
                "feature_columns": feature_columns,
                "medians": medians.fillna(0).to_dict()
            },
            file,
            indent=2
        )

    X = transform_features(
        X_df.to_numpy(dtype=float)
    )

    y = df["label"].to_numpy(dtype=int)

    if len(np.unique(y)) != 2:

        raise RuntimeError(
            "Each client must have both benign and attack samples."
        )

    log.info(
        "Loaded %d training samples with %d features.",
        len(y),
        X.shape[1]
    )

    log.info(
        "Training distribution: benign=%d, attack=%d",
        int(np.sum(y == 0)),
        int(np.sum(y == 1))
    )

    return X, y


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def sigmoid(z):

    z = np.clip(z, -35, 35)

    return 1.0 / (1.0 + np.exp(-z))


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_local_model(
    X,
    y,
    initial_weights,
    initial_bias
):

    weights = initial_weights.copy()

    bias = float(initial_bias)

    counts = np.bincount(y, minlength=2)

    if np.any(counts == 0):

        raise ValueError("Both classes must be present.")

    # Class balancing.
    class_weights = np.array([
        len(y) / (2 * counts[0]),
        len(y) / (2 * counts[1])
    ])

    sample_weights = class_weights[y]

    for epoch in range(LOCAL_EPOCHS):

        probabilities = sigmoid(
            X @ weights + bias
        )

        errors = (
            probabilities - y
        ) * sample_weights

        gradient_weights = (
            X.T @ errors / len(y)
        ) + L2 * weights

        gradient_bias = np.mean(errors)

        weights -= LEARNING_RATE * gradient_weights

        bias -= LEARNING_RATE * gradient_bias

    predictions = (
        sigmoid(X @ weights + bias) >= 0.5
    ).astype(int)

    metrics = {

        "accuracy_train": round(
            float(accuracy_score(y, predictions)), 4
        ),

        "precision_train": round(
            float(precision_score(
                y, predictions, zero_division=0
            )), 4
        ),

        "recall_train": round(
            float(recall_score(
                y, predictions, zero_division=0
            )), 4
        ),

        "f1_train": round(
            float(f1_score(
                y, predictions, zero_division=0
            )), 4
        )
    }

    return weights, bias, metrics


# ============================================================
# WAIT FOR NEXT GLOBAL ROUND
# ============================================================

def wait_for_next_round(previous_round):

    while True:

        response = requests.get(
            f"{SERVER_URL}/model",
            timeout=10
        )

        response.raise_for_status()

        model = response.json()

        if model["round"] > previous_round:

            return model

        time.sleep(2)


# ============================================================
# MAIN FEDERATED TRAINING
# ============================================================

def main():

    wait_for_server()

    X, y = load_local_data()

    log.info(
        "Starting federated training for %s",
        HOME_ID
    )

    # Ensure a fresh experiment.
    while True:

        response = requests.get(
            f"{SERVER_URL}/model",
            timeout=10
        )

        response.raise_for_status()

        server_model = response.json()

        if server_model["round"] == 0:

            break

        time.sleep(2)

    for round_idx in range(1, TRAIN_ROUNDS + 1):

        server_model = requests.get(
            f"{SERVER_URL}/model",
            timeout=10
        ).json()

        global_weights = np.asarray(
            server_model["weights"],
            dtype=float
        )

        global_bias = float(server_model["bias"])

        current_round = server_model["round"]

        log.info(
            "Starting local training for global round %d",
            current_round + 1
        )

        start_time = time.perf_counter()

        local_weights, local_bias, metrics = train_local_model(
            X,
            y,
            global_weights,
            global_bias
        )

        training_seconds = (
            time.perf_counter() - start_time
        )

        payload = {

            "home_id": HOME_ID,

            "weights": local_weights.tolist(),

            "bias": float(local_bias),

            "n_samples": int(len(y)),

            "metrics": metrics,

            "training_seconds": training_seconds
        }

        response = requests.post(
            f"{SERVER_URL}/update",
            json=payload,
            timeout=30
        )

        response.raise_for_status()

        log.info(
            "Submitted update: %s",
            response.json()
        )

        log.info(
            "Training metrics: %s",
            metrics
        )

        log.info(
            "Local training time: %.4f seconds",
            training_seconds
        )

        if round_idx < TRAIN_ROUNDS:

            log.info(
                "Waiting for global round %d",
                current_round + 1
            )

            wait_for_next_round(current_round)

    log.info(
        "Completed %d federated rounds.",
        TRAIN_ROUNDS
    )


if __name__ == "__main__":

    main()