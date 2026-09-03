from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v3_4_0r"
RESULTS = ROOT / "results/v3_4_0r"
V340 = ROOT / "results/v3_4_0"


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
