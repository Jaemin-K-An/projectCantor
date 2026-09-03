"""Two residual attack families with different meanings.

ATTACK-W  ``dh = s * eps * w_hat``  -- along the sensor normal.  Theorem S is
          tight here: ``|dd| = eps`` exactly, so this is the attack the Cantor
          certificate is stated against and the right test of the geometry.

ATTACK-V  ``dh = eps * v_unsafe``   -- along the known causal refusal direction.
          It moves the sensor only by ``eps * |kappa|``, but it is the attack
          that actually changes model behaviour.

They answer different questions and must never be pooled.
"""
from __future__ import annotations

import numpy as np

from .actuator import Actuator
from .sensor_distance import SensorHyperplane


def attack_w(sensor: SensorHyperplane, epsilon: float, sign: int = -1) -> np.ndarray:
    """Sensor-normal perturbation; sign=-1 pushes toward the unsafe side."""
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    return float(sign) * float(epsilon) * sensor.w_hat


def attack_v(actuator: Actuator, epsilon: float) -> np.ndarray:
    """Causal perturbation along the unsafe refusal direction."""
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    return float(epsilon) * actuator.v_unsafe


def expected_delta_d(sensor: SensorHyperplane, delta_h) -> float:
    return float(np.atleast_1d(sensor.delta_distance(delta_h))[0])


def attack_grid_absolute(certificates, *, levels=(0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.01, 1.1, 1.25, 1.5, 2.0)):
    """One ABSOLUTE epsilon grid shared by every rho.

    Anchoring on the family's own certificates guarantees the grid brackets
    every rho's threshold, while the values themselves are identical across
    rho so no controller faces an easier attack than another.
    """
    values = np.asarray(sorted(float(c) for c in certificates), dtype=float)
    if values.size == 0 or np.any(values <= 0):
        raise ValueError("certificates must be positive")
    grid = {0.0}
    for reference in (values.min(), float(np.median(values)), values.max()):
        for level in levels:
            grid.add(round(float(level * reference), 6))
    return sorted(grid)
