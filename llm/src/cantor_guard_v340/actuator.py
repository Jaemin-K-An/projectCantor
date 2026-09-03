"""Frozen causal P0 actuator and its coupling to the behavioural sensor.

The actuator is NOT retrained for V3.4.0.  It is the direction whose causal
effect V3.3.5b/c already replicated; the only permitted operation is to
revalidate it on fresh data.  Rotating ``v`` toward ``w`` after seeing results
would destroy the sensor/actuator test and is forbidden.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cantor_guard_v335c.p0_normalized_dose import normalize_direction

from .sensor_distance import SensorHyperplane


@dataclass(frozen=True)
class Actuator:
    v: np.ndarray
    safe_sign: int

    def __post_init__(self):
        direction = normalize_direction(self.v)
        if int(self.safe_sign) not in (-1, 1):
            raise ValueError("safe_sign must be -1 or +1")
        object.__setattr__(self, "v", direction)
        object.__setattr__(self, "safe_sign", int(self.safe_sign))

    @property
    def v_safe(self) -> np.ndarray:
        """Unit vector whose positive multiples move behaviour toward refusal."""
        return self.safe_sign * self.v

    @property
    def v_unsafe(self) -> np.ndarray:
        return -self.v_safe

    def step(self, h, amplitude):
        """Relative step: ``dh = amplitude * ||h|| * v_safe``."""
        residual = np.asarray(h, dtype=float)
        squeeze = residual.ndim == 1
        batch = residual[None, :] if squeeze else residual
        amp = np.asarray(amplitude, dtype=float).reshape(-1)
        if amp.size == 1:
            amp = np.full(batch.shape[0], float(amp[0]))
        norms = np.linalg.norm(batch, axis=1)
        delta = amp[:, None] * norms[:, None] * self.v_safe[None, :]
        return delta[0] if squeeze else delta


def coupling(sensor: SensorHyperplane, actuator: Actuator) -> dict:
    """kappa and the geometry it implies.

    A controller step ``dh = eta * v_safe`` moves the sensor by exactly
    ``dd = eta * kappa``; nothing about the hypothesis requires kappa to be
    large, but a kappa too small to cross a policy threshold under the frozen
    residual budget makes the controller non-deployable.
    """
    kappa = sensor.coupling(actuator.v_safe)
    cos_wv = sensor.coupling(actuator.v)
    return {
        "kappa": float(kappa),
        "cos_w_v": float(cos_wv),
        "angle_w_v_deg": float(np.degrees(np.arccos(np.clip(cos_wv, -1.0, 1.0)))),
        "sign_kappa": int(np.sign(kappa)),
        "abs_kappa": float(abs(kappa)),
    }


def achievable_sensor_shift(sensor: SensorHyperplane, actuator: Actuator, h_norm, q_max: float):
    """Max |dd| reachable at relative budget q_max: ``q_max*||h||*|kappa|``."""
    norms = np.asarray(h_norm, dtype=float).reshape(-1)
    return float(q_max) * norms * abs(sensor.coupling(actuator.v_safe))
