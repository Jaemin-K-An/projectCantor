from __future__ import annotations

import json
import pathlib

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/v3_3_5c"
RESULTS = ROOT / "results/v3_3_5c"


def read_json(path) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def write_json(path, payload: dict) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n")


def behavioral_protocol() -> dict:
    return read_json(CONFIG / "behavioral_protocol.json")


def load_direction(protocol: dict | None = None) -> np.ndarray:
    protocol = protocol or behavioral_protocol()
    direction = np.load(ROOT / protocol["direction_file"])
    direction = np.asarray(direction, dtype=float).reshape(-1)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid P0 direction")
    return direction / norm


def rho_key(rho: float) -> str:
    return "1/3" if abs(float(rho) - 1 / 3) < 1e-12 else f"{float(rho):.2f}"
