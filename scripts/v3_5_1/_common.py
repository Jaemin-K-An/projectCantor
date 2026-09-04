from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v3_5_1"
RESULTS = ROOT / "results/v3_5_1"
FIGURES = ROOT / "figures/v3_5_1"
BASE_COMMIT = "4c62b9dee2df37f7c775f8b842233d248a34d9b1"
SENSOR_SHA = "f16942ce8c6f89d2eaee2679da4778156450cd44fe1b9ac3529f3434f402f1fe"
ACTUATOR_SHA = "c22957e2fe05e9fa3bc158853dbb5c88965b62a98c2aefd63f11fa73d480172a"
RHOS = (.25, .28, .30, 1/3, .36, .40, .44)
Q_TARGET, Q_CAP, ALPHA, K_RISK = .03, .05, .05, 200


def read_json(path): return json.loads(pathlib.Path(path).read_text())


def write_json(path, payload):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, np.generic): v = v.item()
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    target = pathlib.Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")


def sha256(path): return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def rho_key(rho): return "1/3" if abs(float(rho)-1/3)<1e-12 else f"{float(rho):.2f}"
def arm_key(rho): return f"CANTOR_{rho_key(rho)}"


def frozen_sensor():
    from cantor_guard_v340.sensor_distance import SensorHyperplane
    path = ROOT / "results/v3_4_0/cache/sensor_w.npy"
    if sha256(path) != SENSOR_SHA: raise RuntimeError("frozen sensor hash mismatch")
    b = read_json(ROOT / "results/v3_4_0/tables/sensor_confirm.json")["b"]
    return SensorHyperplane(np.load(path), float(b))


def frozen_actuator():
    from cantor_guard_v340.actuator import Actuator
    path = ROOT / "results/v3_3_5a/cache/v_p0.npy"
    if sha256(path) != ACTUATOR_SHA: raise RuntimeError("frozen actuator hash mismatch")
    return Actuator(np.load(path).astype(float).reshape(-1), 1)


def require_freeze():
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    if freeze.get("status") != "PRE_ANALYSIS_FROZEN":
        raise RuntimeError(f"V3.5.1 final access requires PRE_ANALYSIS_FROZEN, got {freeze.get('status')}")
    if freeze["sensor"]["sha256"] != SENSOR_SHA or freeze["actuator"]["sha256"] != ACTUATOR_SHA:
        raise RuntimeError("frozen hashes changed")
    return freeze


def ensure_final_outputs_absent():
    """Fail closed if a V3.5.1 confirmatory output exists before freeze."""
    forbidden = [
        RESULTS / "raw/final_D_final_v351_harmful.csv",
        RESULTS / "raw/utility_D_final_v351_benign.csv",
        RESULTS / "private/final_D_final_v351_harmful_completions.csv",
        RESULTS / "private/utility_D_final_v351_benign_completions.csv",
    ]
    present = [str(path.relative_to(ROOT)) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"pre-freeze final output access detected: {present}")
    return forbidden
