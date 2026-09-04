"""Depth-3 Cantor geometry applied directly to one-sided risk magnitude."""
from __future__ import annotations

import numpy as np

from cantor_guard_v335c.cantor_geometry import Cell, classify, partition

DEPTH = 3
RHO_CANTOR = 1.0 / 3.0


def margin_m3(rho: float) -> float:
    rho = float(rho)
    if not 0 < rho < 0.5:
        raise ValueError("rho must lie in (0,1/2)")
    return rho**2 * (1.0 - 2.0 * rho)


def margin_derivative_m3(rho: float) -> float:
    rho = float(rho)
    return 2.0 * rho * (1.0 - 3.0 * rho)


def epsilon_r(rho: float, W_R: float) -> float:
    """One-sided residual-L2 terminal-switch certificate ``W_R*M_3``."""
    if not np.isfinite(W_R) or W_R <= 0:
        raise ValueError("W_R must be finite and positive")
    return float(W_R) * margin_m3(rho)


def epsilon_r_cantor(W_R: float) -> float:
    return float(W_R) / 27.0


def unique_grid_max(rhos) -> float:
    """Return the rho with largest certificate factor, rejecting ties."""
    arr = np.asarray(rhos, dtype=float)
    values = np.asarray([margin_m3(r) for r in arr])
    winners = np.flatnonzero(np.isclose(values, values.max(), atol=1e-15, rtol=0))
    if winners.size != 1:
        raise ValueError("rho grid does not have a unique maximum")
    return float(arr[winners[0]])


__all__ = ["Cell", "DEPTH", "RHO_CANTOR", "classify", "partition",
           "margin_m3", "margin_derivative_m3", "epsilon_r",
           "epsilon_r_cantor", "unique_grid_max"]
