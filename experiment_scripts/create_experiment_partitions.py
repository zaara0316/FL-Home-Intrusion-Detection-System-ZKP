import os
import json
import shutil

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "/app"

ORIGINAL_PARTITIONS = os.path.join(
    BASE_DIR,
    "dataset",
    "partitions"
)

EXPERIMENT_DIR = os.path.join(
    BASE_DIR,
    "experiments"
)

RANDOM_STATE = 42

CLIENTS = 5

# Exact benign allocation for non-IID experiment.
NON_IID_BENIGN_COUNTS = [
    100,
    300,
    700,
    1500,
    3000
]


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():

    for experiment in ["iid", "non_iid"]:

        partition_dir = os.path.join(
            EXPERIMENT_DIR,
            experiment,
            "partitions"
        )

        os.makedirs(
            partition_dir,
            exist_ok=True
        )


# ============================================================
# LOAD ORIGINAL TRAINING AND TEST DATA
# ============================================================

def load_original_data():

    train_frames = []
    test_frames = []

    for home in range(1, CLIENTS + 1):

        train_path = os.path.join(
            ORIGINAL_PARTITIONS,
            f"home-{home}-train.csv"
        )

        test_path = os.path.join(
            ORIGINAL_PARTITIONS,
            f"home-{home}-test.csv"
        )

        if not os.path.exists(train_path):
            raise FileNotFoundError(train_path)

        if not os.path.exists(test_path):
            raise FileNotFoundError(test_path)

        train_frames.append(
            pd.read_csv(train_path)
        )

        test_frames.append(
            pd.read_csv(test_path)
        )

    train_df = pd.concat(
        train_frames,
        ignore_index=True
    )

    test_df = pd.concat(
        test_frames,
        ignore_index=True
    )

    return train_df, test_df


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(train_df, test_df):

    required = {
        "device_id",
        "label",
        "attack_type"
    }

    if not required.issubset(train_df.columns):
        raise ValueError("Required training metadata missing.")

    if not required.issubset(test_df.columns):
        raise ValueError("Required test metadata missing.")

    if len(train_df) != 64800:
        raise ValueError(
            f"Expected 64800 training samples, got {len(train_df)}"
        )

    if len(test_df) != 16200:
        raise ValueError(
            f"Expected 16200 test samples, got {len(test_df)}"
        )

    counts = train_df["label"].value_counts().to_dict()

    if counts.get(0, 0) != 5600:
        raise ValueError("Unexpected benign sample count.")

    if counts.get(1, 0) != 59200:
        raise ValueError("Unexpected attack sample count.")

    print("\nTraining data validated.")

    print("Training samples:", len(train_df))
    print("Testing samples:", len(test_df))

    print("Training label distribution:", counts)


# ============================================================
# SPLIT A CLASS INTO FIVE CLIENTS
# ============================================================

def split_class(df, counts, rng):

    shuffled = df.sample(
        frac=1,
        random_state=int(
            rng.integers(0, 2**31 - 1)
        )
    ).reset_index(drop=True)

    expected = sum(counts)

    if len(shuffled) != expected:
        raise ValueError(
            "Class allocation does not match available samples."
        )

    result = []

    start = 0

    for count in counts:

        end = start + count

        result.append(
            shuffled.iloc[start:end].copy()
        )

        start = end

    return result


# ============================================================
# CREATE IID PARTITIONS
# ============================================================

def create_iid_partitions(train_df, test_df):

    print("\n" + "=" * 65)
    print("CREATING IID PARTITIONS")
    print("=" * 65)

    rng = np.random.default_rng(RANDOM_STATE)

    benign = train_df[
        train_df["label"] == 0
    ]

    attacks = train_df[
        train_df["label"] == 1
    ]

    benign_counts = [1120] * 5
    attack_counts = [11840] * 5

    benign_splits = split_class(
        benign,
        benign_counts,
        rng
    )

    attack_splits = split_class(
        attacks,
        attack_counts,
        rng
    )

    save_partitions(
        "iid",
        benign_splits,
        attack_splits,
        test_df,
        rng
    )


# ============================================================
# CREATE NON-IID PARTITIONS
# ============================================================

def create_non_iid_partitions(train_df, test_df):

    print("\n" + "=" * 65)
    print("CREATING NON-IID PARTITIONS")
    print("=" * 65)

    rng = np.random.default_rng(
        RANDOM_STATE + 1
    )

    benign = train_df[
        train_df["label"] == 0
    ]

    attacks = train_df[
        train_df["label"] == 1
    ]

    benign_counts = NON_IID_BENIGN_COUNTS

    attack_counts = [
        12960 - count
        for count in benign_counts
    ]

    benign_splits = split_class(
        benign,
        benign_counts,
        rng
    )

    attack_splits = split_class(
        attacks,
        attack_counts,
        rng
    )

    save_partitions(
        "non_iid",
        benign_splits,
        attack_splits,
        test_df,
        rng
    )


# ============================================================
# SAVE PARTITIONS
# ============================================================

def save_partitions(
    experiment,
    benign_splits,
    attack_splits,
    test_df,
    rng
):

    output_dir = os.path.join(
        EXPERIMENT_DIR,
        experiment,
        "partitions"
    )

    manifest = {
        "experiment": experiment,
        "random_state": RANDOM_STATE,
        "clients": []
    }

    for i in range(CLIENTS):

        client_df = pd.concat(
            [
                benign_splits[i],
                attack_splits[i]
            ],
            ignore_index=True
        )

        client_df = client_df.sample(
            frac=1,
            random_state=int(
                rng.integers(0, 2**31 - 1)
            )
        ).reset_index(drop=True)

        if client_df["label"].nunique() != 2:
            raise ValueError(
                f"Client {i+1} does not have both classes."
            )

        train_path = os.path.join(
            output_dir,
            f"home-{i+1}-train.csv"
        )

        client_df.to_csv(
            train_path,
            index=False
        )

        benign_count = int(
            (client_df["label"] == 0).sum()
        )

        attack_count = int(
            (client_df["label"] == 1).sum()
        )

        client_info = {

            "home_id": f"home-{i+1}",

            "total_samples": len(client_df),

            "benign_samples": benign_count,

            "attack_samples": attack_count,

            "benign_proportion": (
                benign_count / len(client_df)
            ),

            "attack_proportion": (
                attack_count / len(client_df)
            ),

            "device_distribution": {
                str(k): int(v)
                for k, v in client_df[
                    "device_id"
                ].value_counts().items()
            }

        }

        manifest["clients"].append(
            client_info
        )

        print(
            f"\nHome {i+1}: "
            f"total={len(client_df)}, "
            f"benign={benign_count}, "
            f"attack={attack_count}"
        )

        print(
            "Device distribution:",
            client_info["device_distribution"]
        )

    # Save the same fixed test partitions for both experiments.
    for i in range(CLIENTS):

        device_id = i + 1

        original_test = pd.read_csv(
            os.path.join(
                ORIGINAL_PARTITIONS,
                f"home-{device_id}-test.csv"
            )
        )

        test_path = os.path.join(
            output_dir,
            f"home-{device_id}-test.csv"
        )

        original_test.to_csv(
            test_path,
            index=False
        )

    manifest_path = os.path.join(
        EXPERIMENT_DIR,
        experiment,
        "partition_manifest.json"
    )

    with open(manifest_path, "w") as file:

        json.dump(
            manifest,
            file,
            indent=4
        )

    print(
        f"\nSaved {experiment} partition manifest."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("FEDNATD CONTROLLED DATA PARTITIONING")
    print("=" * 65)

    create_directories()

    train_df, test_df = load_original_data()

    validate_data(
        train_df,
        test_df
    )

    create_iid_partitions(
        train_df,
        test_df
    )

    create_non_iid_partitions(
        train_df,
        test_df
    )

    print("\n" + "=" * 65)
    print("PARTITION GENERATION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":

    main()