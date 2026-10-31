import numpy as np

DEVICE_PROFILES = {
    "doorbell": {"offset": 0.1, "rate": 0.8},
    "thermostat": {"offset": 0.3, "rate": 0.4},
    "security_camera": {"offset": 0.6, "rate": 1.0},
    "smart_speaker": {"offset": 0.2, "rate": 0.7},
    "motion_sensor": {"offset": 0.4, "rate": 0.3},
    "smart_plug": {"offset": 0.5, "rate": 0.5},
}

def generate_device_samples(device_name, n_samples, n_features, seed):
    """Synthetic smoke-test rows only; not real network traffic or research data."""
    rng = np.random.default_rng(seed)
    profile = DEVICE_PROFILES.get(device_name, {"offset": 0.25, "rate": 0.5})
    X = rng.normal(profile["offset"], 0.25, size=(n_samples, n_features))
    p_attack = min(0.35, 0.04 + profile["rate"] * 0.08)
    y = (rng.random(n_samples) < p_attack).astype(int)
    if y.any():
        X[y == 1, :min(3, n_features)] += 1.25
    return X, y
