import numpy as np
import pandas as pd
from pathlib import Path

# Dataset configuration
DATASET_DIR = Path("/dataset")
OUTPUT_DIR = DATASET_DIR / "processed"

CHUNK_SIZE = 25000
SAMPLES_PER_CHUNK = 500
MAX_SAMPLES_PER_FILE = 3000
RANDOM_SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_labels(filename):
    """
    Extract device ID, attack type and binary label
    from the CSV filename.

    Example:
    1.benign.csv -> device 1, benign, label 0
    1.mirai.udp.csv -> device 1, mirai.udp, label 1
    """

    parts = filename.removesuffix(".csv").split(".")

    device_id = parts[0]
    traffic_type = ".".join(parts[1:])

    label = 0 if traffic_type == "benign" else 1

    return device_id, traffic_type, label


def sample_csv(path):
    """
    Read a large CSV in chunks and sample rows
    to reduce memory consumption.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    sampled_chunks = []

    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE):

        chunk = chunk.apply(pd.to_numeric, errors="coerce")

        chunk = chunk.replace(
            [np.inf, -np.inf],
            np.nan
        )

        chunk = chunk.dropna(how="all")

        if chunk.empty:
            continue

        sample_count = min(
            SAMPLES_PER_CHUNK,
            len(chunk)
        )

        sampled = chunk.sample(
            n=sample_count,
            random_state=int(
                rng.integers(0, 2**31 - 1)
            )
        )

        sampled_chunks.append(sampled)

    if not sampled_chunks:
        return pd.DataFrame()

    result = pd.concat(
        sampled_chunks,
        ignore_index=True
    )

    if len(result) > MAX_SAMPLES_PER_FILE:

        result = result.sample(
            n=MAX_SAMPLES_PER_FILE,
            random_state=RANDOM_SEED
        )

    return result.reset_index(drop=True)


def main():

    csv_files = sorted(
        path for path in DATASET_DIR.glob("*.csv")
        if path.is_file()
    )

    if not csv_files:
        raise RuntimeError(
            "No CSV files found in /dataset"
        )

    print(f"Found {len(csv_files)} CSV files.")

    all_samples = []
    expected_columns = None

    for index, path in enumerate(csv_files, start=1):

        device_id, traffic_type, label = get_labels(
            path.name
        )

        print(
            f"[{index}/{len(csv_files)}] Processing {path.name}"
        )

        sample = sample_csv(path)

        if sample.empty:
            print(f"Skipping empty file: {path.name}")
            continue

        if expected_columns is None:

            expected_columns = list(sample.columns)

        elif list(sample.columns) != expected_columns:

            raise ValueError(
                f"Feature mismatch in {path.name}"
            )

        sample["device_id"] = device_id
        sample["attack_type"] = traffic_type
        sample["label"] = label

        all_samples.append(sample)

        print(
            f"  Sampled {len(sample)} rows | "
            f"Attack type: {traffic_type} | "
            f"Label: {label}"
        )

    if not all_samples:
        raise RuntimeError(
            "No usable samples were collected."
        )

    print("\nCombining sampled data...")

    dataset = pd.concat(
        all_samples,
        ignore_index=True
    )

    feature_columns = expected_columns

    # Keep missing values for now.
    # Imputation will happen inside the ML pipeline
    # after the train/test split to avoid data leakage.

    dataset[feature_columns] = dataset[
        feature_columns
    ].replace([np.inf, -np.inf], np.nan)

    # Remove features that contain no usable values.
    usable_features = [
        column for column in feature_columns
        if dataset[column].notna().any()
    ]

    dataset = dataset[
        usable_features + [
            "device_id",
            "attack_type",
            "label"
        ]
    ]

    dataset = dataset.dropna(
        subset=usable_features,
        how="all"
    )

    output_path = OUTPUT_DIR / "nbaiot_sample.csv"

    dataset.to_csv(
        output_path,
        index=False
    )

    print("\n========== PREPROCESSING COMPLETE ==========")

    print(f"Output file: {output_path}")
    print(f"Total samples: {len(dataset)}")
    print(f"Number of features: {len(usable_features)}")

    print("\nBinary label distribution:")
    print(dataset["label"].value_counts())

    print("\nAttack type distribution:")
    print(dataset["attack_type"].value_counts())

    print("\nDevice distribution:")
    print(dataset["device_id"].value_counts())

    print("\nMissing values:")
    print(dataset[usable_features].isna().sum().sum())

    print("=============================================")


if __name__ == "__main__":
    main()