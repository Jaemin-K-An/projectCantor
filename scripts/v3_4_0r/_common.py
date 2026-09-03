from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v3_4_0r"
RESULTS = ROOT / "results/v3_4_0r"
V340 = ROOT / "results/v3_4_0"
FROZEN_W = 2.2805212277347544
Q_TARGET = 0.03
Q_CAP = 0.05


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


def rho_key(rho: float) -> str:
    return "1/3" if abs(float(rho) - 1 / 3) < 1e-12 else f"{float(rho):.2f}"


def frozen_sensor():
    from cantor_guard_v340.sensor_distance import SensorHyperplane

    fit = read_json(V340 / "tables" / "sensor_confirm.json")
    return SensorHyperplane(np.load(V340 / "cache" / "sensor_w.npy"), float(fit["b"]))


def frozen_actuator():
    from cantor_guard_v340.actuator import Actuator

    cfg = read_json(ROOT / "configs/v3_4_0/actuator.json")
    return Actuator(np.load(ROOT / cfg["direction_file"]).astype(float).reshape(-1),
                    int(cfg["safe_sign"]))


def require_confirmatory_freeze() -> dict:
    """Load a valid freeze or stop before any final/certificate operation."""
    manifest = read_json(CONFIG / "PRE_ANALYSIS_FREEZE.json")
    if manifest.get("status") != "PRE_ANALYSIS_FROZEN":
        raise RuntimeError(
            "V3.4.0R final-stage execution requires PRE_ANALYSIS_FROZEN; "
            f"observed {manifest.get('status', 'MISSING')}"
        )
    if float(manifest["inherited_frozen"]["W"]) != FROZEN_W:
        raise RuntimeError("frozen W mismatch")
    if float(manifest["budget"]["q_target_rms"]) != Q_TARGET:
        raise RuntimeError("frozen q target mismatch")
    if float(manifest["hard_q_cap"]["q_cap"]) != Q_CAP:
        raise RuntimeError("frozen q cap mismatch")
    return manifest


def require_external_window_pass() -> None:
    transfer = read_json(RESULTS / "tables/sensor_transfer.json")
    window = read_json(RESULTS / "tables/external_window.json")
    if transfer.get("transport_verdict") != "ST1_PASS":
        raise RuntimeError("external sensor transport gate failed")
    if window.get("verdict") != "ST1_PASS":
        raise RuntimeError("external fixed-W applicability gate failed")
