"""Frozen depth-3 symmetric Cantor policy geometry on the sensor coordinate.

Identical geometry to V3.3.5c; only the coordinate it is imposed on changes
(signed sensor distance instead of actuator projection).  The certificate is
therefore renamed ``epsilon_h`` to signal that it bounds a residual-space L2
perturbation directly, via Theorem S.
"""
from __future__ import annotations

from cantor_guard_v335c.cantor_geometry import (  # noqa: F401
    DEPTH,
    RHO_CANTOR,
    Cell,
    classify,
    direct_terminal_transition,
    margin_derivative_m3,
    margin_m3,
    partition,
)


def epsilon_h(rho, W: float):
    """Sufficient residual-L2 radius against direct terminal-policy switching.

    ``|dr| <= ||dh||/(2W)`` (Theorem S plus the affine map), and a direct
    terminal switch needs ``|dr| >= M_3(rho)``; hence ``||dh|| < 2W M_3(rho)``
    is sufficient.  Maximised uniquely at rho=1/3 where it equals 2W/27.
    """
    if float(W) <= 0:
        raise ValueError("W must be positive")
    return 2 * float(W) * margin_m3(rho)


def epsilon_h_cantor(W: float) -> float:
    return 2 * float(W) / 27
