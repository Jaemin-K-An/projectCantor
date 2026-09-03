"""Frozen symmetric depth-3 Cantor-family policy geometry and certificate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RHO_CANTOR = 1 / 3
DEPTH = 3


@dataclass(frozen=True)
class Cell:
    kind: str
    lo: float
    hi: float
    index: int | None = None
    level: int | None = None
    address: str | None = None


def _validate(rho: float, depth: int) -> None:
    if not 0 < float(rho) < 0.5:
        raise ValueError("rho must lie in (0, 1/2)")
    if depth < 1:
        raise ValueError("depth must be >= 1")


def partition(rho: float, depth: int = DEPTH) -> tuple[list[Cell], list[Cell]]:
    """Return ordered terminal leaves and all guards in closed form."""
    _validate(rho, depth)
    leaves: list[Cell] = []
    guards: list[Cell] = []

    def recurse(lo: float, hi: float, level: int, address: str) -> None:
        span = hi - lo
        left_hi = lo + rho * span
        right_lo = hi - rho * span
        guards.append(Cell("guard", left_hi, right_lo, level=level, address=address))
        if level == depth:
            leaves.append(Cell("leaf", lo, left_hi, address=address + "0"))
            leaves.append(Cell("leaf", right_lo, hi, address=address + "1"))
            return
        recurse(lo, left_hi, level + 1, address + "0")
        recurse(right_lo, hi, level + 1, address + "1")

    recurse(0.0, 1.0, 1, "")
    leaves.sort(key=lambda c: c.lo)
    leaves = [Cell(c.kind, c.lo, c.hi, index=i, address=c.address) for i, c in enumerate(leaves)]
    guards.sort(key=lambda c: (c.lo, c.hi))
    return leaves, guards


def classify(r: float, rho: float, depth: int = DEPTH) -> Cell:
    if not np.isfinite(r) or r < 0 or r > 1:
        return Cell("outside", float("nan"), float("nan"))
    leaves, guards = partition(rho, depth)
    # Boundaries belong to guards when a guard touches them; this makes the
    # conservative policy explicit and avoids boundary-dependent leaf switches.
    for guard in guards:
        if guard.lo <= r <= guard.hi:
            return guard
    for leaf in leaves:
        if leaf.lo <= r <= leaf.hi:
            return leaf
    raise RuntimeError("partition failed to cover [0,1]")


def margin_m3(rho):
    value = np.asarray(rho, dtype=float)
    return value**2 * (1 - 2 * value)


def margin_derivative_m3(rho):
    value = np.asarray(rho, dtype=float)
    return 2 * value * (1 - 3 * value)


def epsilon_z(rho, W: float):
    if W <= 0:
        raise ValueError("W must be positive")
    return 2 * float(W) * margin_m3(rho)


def epsilon_cantor(W: float) -> float:
    return 2 * float(W) / 27


def direct_terminal_transition(clean_r: float, attacked_r: float, rho: float, depth: int = DEPTH) -> bool:
    clean = classify(clean_r, rho, depth)
    attacked = classify(attacked_r, rho, depth)
    return clean.kind == attacked.kind == "leaf" and clean.index != attacked.index
