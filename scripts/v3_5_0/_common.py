from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v3_5_0"
RESULTS = ROOT / "results/v3_5_0"
FIGURES = ROOT / "figures/v3_5_0"
V340 = ROOT / "results/v3_4_0"
BASE_COMMIT = "8d04b67933aa95b1fc452b27d6a2c77517486332"
SENSOR_SHA256 = "f16942ce8c6f89d2eaee2679da4778156450cd44fe1b9ac3529f3434f402f1fe"
ACTUATOR_SHA256 = "c22957e2fe05e9fa3bc158853dbb5c88965b62a98c2aefd63f11fa73d480172a"
Q_TARGET = 0.03
Q_CAP = 0.05
ALPHA = 0.05
RHOS = (0.25, 0.28, 0.30, 1 / 3, 0.36, 0.40, 0.44)


def read_json(path) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def write_json(path, payload: dict) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")


def sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def rho_key(rho: float) -> str:
    return "1/3" if abs(float(rho) - 1 / 3) < 1e-12 else f"{float(rho):.2f}"


def frozen_sensor():
    from cantor_guard_v340.sensor_distance import SensorHyperplane
    fit = read_json(V340 / "tables/sensor_confirm.json")
    path = V340 / "cache/sensor_w.npy"
    if sha256(path) != SENSOR_SHA256:
        raise RuntimeError("frozen sensor hash mismatch")
    return SensorHyperplane(np.load(path), float(fit["b"]))


def frozen_actuator():
    from cantor_guard_v340.actuator import Actuator
    cfg = read_json(ROOT / "configs/v3_4_0/actuator.json")
    path = ROOT / cfg["direction_file"]
    if sha256(path) != ACTUATOR_SHA256:
        raise RuntimeError("frozen actuator hash mismatch")
    return Actuator(np.load(path).astype(float).reshape(-1), int(cfg["safe_sign"]))


def require_freeze() -> dict:
    freeze = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    if freeze.get("status") != "PRE_ANALYSIS_FROZEN":
        raise RuntimeError(f"V3.5.0 final operation requires PRE_ANALYSIS_FROZEN, got {freeze.get('status')}")
    if freeze["sensor"]["sha256"] != SENSOR_SHA256 or freeze["actuator"]["sha256"] != ACTUATOR_SHA256:
        raise RuntimeError("frozen sensor/actuator hash mismatch")
    if float(freeze["budget"]["q_target_rms"]) != Q_TARGET:
        raise RuntimeError("q_target must remain .03")
    if float(freeze["budget"]["q_cap"]) != Q_CAP:
        raise RuntimeError("q_cap must remain .05")
    return freeze


def ensure_final_absent() -> None:
    paths = [RESULTS / "raw/final_D_final_v350_harmful.csv",
             RESULTS / "raw/utility_D_final_v350_benign.csv"]
    if any(p.exists() for p in paths):
        raise RuntimeError("final output already exists; prospective calibration cannot be changed")


def build_arms(freeze, sensor, actuator):
    from cantor_guard_v350.linear_risk_controller import LinearRiskController
    from cantor_guard_v350.risk_cantor_controller import RiskCantorController
    W_R = float(freeze["risk_window"]["W_R"])
    q_cap = float(freeze["budget"]["q_cap"])
    eta = freeze["budget"]["eta_per_arm"]
    arms = {"ATTACK_ONLY": None,
            "LINEAR": LinearRiskController(sensor=sensor, actuator=actuator, W_R=W_R,
                                            eta=float(eta["LINEAR"]), q_cap=q_cap)}
    for rho in freeze["geometry"]["rho_family"]:
        key = rho_key(rho)
        arms[key] = RiskCantorController(
            sensor=sensor, actuator=actuator, W_R=W_R, rho=float(rho),
            eta=float(eta[key]), q_cap=q_cap,
            leaf_actions=freeze["geometry"]["leaf_actions"],
            outside_action=float(freeze["geometry"]["outside_risk_action"]),
        )
    return arms
