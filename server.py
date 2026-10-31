import os
import json
import time
import threading

from pathlib import Path

import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import uvicorn


# ============================================================
# CONFIGURATION
# ============================================================

HOST = os.getenv("SERVER_HOST", "0.0.0.0")

PORT = int(os.getenv("SERVER_PORT", "8000"))

N_FEATURES = int(os.getenv("MODEL_FEATURES", "115"))

CLIENT_IDS = {
    "home-1",
    "home-2",
    "home-3",
    "home-4",
    "home-5"
}

TOTAL_CLIENTS = len(CLIENT_IDS)

ARTIFACT_DIR = Path("/app/artifacts")

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = ARTIFACT_DIR / "federated_global_model.json"

HISTORY_PATH = ARTIFACT_DIR / "fedavg_history.json"


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="FedNatD Federated Learning Server"
)

lock = threading.Lock()


global_model = {

    "weights": np.zeros(N_FEATURES, dtype=float),

    "bias": 0.0,

    "round": 0
}

client_updates = {}

history = []


# ============================================================
# REQUEST MODEL
# ============================================================

class ModelUpdate(BaseModel):

    home_id: str

    weights: list[float]

    bias: float

    n_samples: int

    metrics: dict = Field(default_factory=dict)

    training_seconds: float = 0.0


# ============================================================
# PERSISTENCE
# ============================================================

def save_experiment():

    model_data = {

        "weights": global_model["weights"].tolist(),

        "bias": float(global_model["bias"]),

        "round": global_model["round"]
    }

    temporary_model = MODEL_PATH.with_suffix(".tmp")

    with open(temporary_model, "w") as file:

        json.dump(model_data, file, indent=2)

    os.replace(
        temporary_model,
        MODEL_PATH
    )

    temporary_history = HISTORY_PATH.with_suffix(".tmp")

    with open(temporary_history, "w") as file:

        json.dump(
            history,
            file,
            indent=2
        )

    os.replace(
        temporary_history,
        HISTORY_PATH
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    with lock:

        return {

            "status": "ok",

            "round": global_model["round"],

            "received_updates": len(client_updates),

            "total_clients": TOTAL_CLIENTS
        }


# ============================================================
# GLOBAL MODEL
# ============================================================

@app.get("/model")
def get_model():

    with lock:

        return {

            "weights": global_model["weights"].tolist(),

            "bias": float(global_model["bias"]),

            "round": global_model["round"]
        }


# ============================================================
# STATUS
# ============================================================

@app.get("/status")
def get_status():

    with lock:

        return {

            "round": global_model["round"],

            "received_updates": len(client_updates),

            "total_clients": TOTAL_CLIENTS,

            "completed": len(client_updates) == 0
        }


# ============================================================
# FEDAVG UPDATE
# ============================================================

@app.post("/update")
def update_model(update: ModelUpdate):

    if update.home_id not in CLIENT_IDS:

        raise HTTPException(
            status_code=400,
            detail="Unknown home ID"
        )

    if len(update.weights) != N_FEATURES:

        raise HTTPException(
            status_code=400,
            detail=f"Expected {N_FEATURES} weights"
        )

    if update.n_samples < 1:

        raise HTTPException(
            status_code=400,
            detail="n_samples must be positive"
        )

    if update.training_seconds < 0:

        raise HTTPException(
            status_code=400,
            detail="Training time cannot be negative"
        )

    weights = np.asarray(
        update.weights,
        dtype=float
    )

    if (
        not np.all(np.isfinite(weights))
        or not np.isfinite(update.bias)
    ):

        raise HTTPException(
            status_code=400,
            detail="Model parameters must be finite"
        )

    with lock:

        if update.home_id in client_updates:

            raise HTTPException(
                status_code=409,
                detail="Client already submitted for this round"
            )

        if len(client_updates) >= TOTAL_CLIENTS:

            raise HTTPException(
                status_code=409,
                detail="Current round is already complete"
            )

        client_updates[update.home_id] = {

            "weights": weights,

            "bias": float(update.bias),

            "n_samples": update.n_samples,

            "metrics": update.metrics,

            "training_seconds": update.training_seconds,

            "update_bytes": len(
                json.dumps(
                    update.model_dump()
                ).encode("utf-8")
            )
        }

        print(
            f"[server] Received update from {update.home_id} "
            f"({len(client_updates)}/{TOTAL_CLIENTS})",
            flush=True
        )

        aggregated = False

        if len(client_updates) == TOTAL_CLIENTS:

            aggregation_start = time.perf_counter()

            total_samples = sum(
                item["n_samples"]
                for item in client_updates.values()
            )

            new_weights = np.zeros(
                N_FEATURES,
                dtype=float
            )

            new_bias = 0.0

            for home_id, item in client_updates.items():

                contribution = (
                    item["n_samples"] / total_samples
                )

                new_weights += (
                    contribution * item["weights"]
                )

                new_bias += (
                    contribution * item["bias"]
                )

                print(
                    f"[server] {home_id}: "
                    f"samples={item['n_samples']}, "
                    f"weight={contribution:.4f}",
                    flush=True
                )

            global_model["weights"] = new_weights

            global_model["bias"] = float(new_bias)

            global_model["round"] += 1

            aggregation_seconds = (
                time.perf_counter() - aggregation_start
            )

            round_record = {

                "round": global_model["round"],

                "total_samples": total_samples,

                "participating_clients": TOTAL_CLIENTS,

                "client_metrics": {

                    home_id: {

                        "training_metrics": item["metrics"],

                        "training_seconds": item["training_seconds"],

                        "update_bytes": item["update_bytes"],

                        "sample_count": item["n_samples"]

                    }

                    for home_id, item in client_updates.items()
                },

                "aggregation_seconds": aggregation_seconds,

                "total_update_bytes": sum(
                    item["update_bytes"]
                    for item in client_updates.values()
                )
            }

            history.append(round_record)

            save_experiment()

            print(
                f"\n[server] FedAvg aggregation completed. "
                f"Global round={global_model['round']}",
                flush=True
            )

            print(
                f"[server] Aggregation time: "
                f"{aggregation_seconds:.6f} seconds",
                flush=True
            )

            client_updates.clear()

            aggregated = True

        return {

            "accepted": True,

            "aggregated": aggregated,

            "round": global_model["round"],

            "received_updates": len(client_updates)
        }


# ============================================================
# HISTORY
# ============================================================

@app.get("/history")
def get_history():

    with lock:

        return {

            "history": history
        }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host=HOST,
        port=PORT
    )