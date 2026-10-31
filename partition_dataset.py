import os
import pandas as pd

from sklearn.model_selection import train_test_split

# Configuration
INPUT_FILE = "/dataset/processed/nbaiot_sample.csv"
OUTPUT_DIR = "/app/dataset/partitions"

TEST_SIZE = 0.20
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_partitions():

    print("=" * 60)
    print("FEDNATD DATASET PARTITIONING")
    print("=" * 60)

    # Load processed dataset
    print("\nLoading processed N-BaIoT dataset...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Processed dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Total samples: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    if "device_id" not in df.columns:
        raise ValueError("device_id column missing")

    if "label" not in df.columns:
        raise ValueError("label column missing")

    print("\nOverall label distribution:")
    print(df["label"].value_counts())

    # Generate partitions for five homes
    for device_id in range(1, 6):

        print("\n" + "-" * 60)
        print(f"Processing Home {device_id}")
        print("-" * 60)

        device_df = df[
            df["device_id"].astype(str) == str(device_id)
        ].copy()

        if device_df.empty:
            raise ValueError(
                f"No samples found for device {device_id}"
            )

        print(f"Total device samples: {len(device_df)}")

        print("\nDevice label distribution:")
        print(device_df["label"].value_counts())

        # Separate features and labels
        X = device_df.drop(columns=["label"])
        y = device_df["label"]

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        # Recombine metadata, features and labels
        train_df = X_train.copy()
        train_df["label"] = y_train

        test_df = X_test.copy()
        test_df["label"] = y_test

        # Save files
        train_path = os.path.join(
            OUTPUT_DIR,
            f"home-{device_id}-train.csv"
        )

        test_path = os.path.join(
            OUTPUT_DIR,
            f"home-{device_id}-test.csv"
        )

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\nTraining samples: {len(train_df)}")
        print(f"Testing samples: {len(test_df)}")

        print("Training labels:")
        print(train_df["label"].value_counts().to_dict())

        print("Testing labels:")
        print(test_df["label"].value_counts().to_dict())

        print(f"Saved: {train_path}")
        print(f"Saved: {test_path}")

    print("\n" + "=" * 60)
    print("PARTITIONING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    create_partitions()