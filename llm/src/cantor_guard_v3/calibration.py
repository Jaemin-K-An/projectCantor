"""V3 PHASE 4 — dynamic calibration of the safety boundary.

PHASE 3 measured that tau moves by a median of 0.34 sigma and up to 1.77 sigma
across phase / token position / attack family. V2 used ONE tau per layer,
fitted on the last prompt token. This module implements the alternatives and
lets them be compared head to head with geometry (harness §38): if simply
estimating tau better beats every controller geometry, that is the finding.

  C0 fixed        one (tau, sigma) per layer               [what V2 did]
  C1 phase        separate for prefill vs generation
  C2 token-bin    separate per generation-position bin
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

BINS = ("prefill", "gen_1", "gen_2-4", "gen_5-8", "gen_9-16", "gen_17+")


def bin_of(pos: int) -> str:
    if pos == 0: return "prefill"
    if pos == 1: return "gen_1"
    if pos <= 4: return "gen_2-4"
    if pos <= 8: return "gen_5-8"
    if pos <= 16: return "gen_9-16"
    return "gen_17+"


def phase_of(pos: int) -> str:
    return "prefill" if pos == 0 else "generation"


@dataclass
class Calibrator:
    """Maps (layer, generation position) -> (tau, sigma).

    `method` in {"C0_fixed", "C1_phase", "C2_token_bin"}. Falls back to the
    layer-global value whenever a bin was not populated at fit time, so a
    dynamic calibrator can never be worse-specified than C0.
    """
    method: str
    tau_global: dict = field(default_factory=dict)     # layer -> tau
    sigma_global: dict = field(default_factory=dict)
    tau_cond: dict = field(default_factory=dict)       # (layer, key) -> tau
    sigma_cond: dict = field(default_factory=dict)

    def key(self, pos: int) -> str:
        if self.method == "C1_phase":
            return phase_of(pos)
        if self.method == "C2_token_bin":
            return bin_of(pos)
        return "global"

    def get(self, layer: int, pos: int) -> tuple[float, float]:
        if self.method == "C0_fixed":
            return self.tau_global[layer], self.sigma_global[layer]
        k = (layer, self.key(pos))
        if k in self.tau_cond:
            return self.tau_cond[k], self.sigma_cond[k]
        return self.tau_global[layer], self.sigma_global[layer]

    def margin(self, z: np.ndarray, layer: int, pos: int) -> np.ndarray:
        tau, sig = self.get(layer, pos)
        return (np.asarray(z, dtype=float) - tau) / sig


def fit_calibrator(proj_df, method: str) -> Calibrator:
    """Fit from a long-form projection frame with columns
    layer, pos, z, cls ('harmful'/'harmless')."""
    c = Calibrator(method=method)
    for l, gl in proj_df.groupby("layer"):
        zh = gl[gl.cls == "harmful"].z.values
        zb = gl[gl.cls == "harmless"].z.values
        c.tau_global[int(l)] = 0.5 * (zh.mean() + zb.mean())
        c.sigma_global[int(l)] = float(np.sqrt(0.5*(zh.var()+zb.var()))) + 1e-8
    if method == "C0_fixed":
        return c
    keyfn = phase_of if method == "C1_phase" else bin_of
    for l, gl in proj_df.groupby("layer"):
        g2 = gl.assign(_k=gl.pos.map(keyfn))
        for k, sub in g2.groupby("_k"):
            zh = sub[sub.cls == "harmful"].z.values
            zb = sub[sub.cls == "harmless"].z.values
            if len(zh) < 5 or len(zb) < 5:
                continue
            c.tau_cond[(int(l), k)] = 0.5 * (zh.mean() + zb.mean())
            c.sigma_cond[(int(l), k)] = float(np.sqrt(0.5*(zh.var()+zb.var()))) + 1e-8
    return c


def calibration_error(proj_df, cal: Calibrator) -> float:
    """Mean |residual boundary offset| in sigma units on held-out projections.

    For each (layer, bin) group, the ideal threshold is the local midpoint; the
    error is how far the calibrator's tau sits from it. This is exactly the
    Delta the controller suffers.
    """
    errs = []
    for (l, k), sub in proj_df.assign(_k=proj_df.pos.map(bin_of)).groupby(["layer", "_k"]):
        zh = sub[sub.cls == "harmful"].z.values
        zb = sub[sub.cls == "harmless"].z.values
        if len(zh) < 5 or len(zb) < 5:
            continue
        ideal = 0.5 * (zh.mean() + zb.mean())
        pos = {"prefill": 0, "gen_1": 1, "gen_2-4": 3, "gen_5-8": 6,
               "gen_9-16": 12, "gen_17+": 20}[k]
        tau, _ = cal.get(int(l), pos)
        errs.append(abs(ideal - tau) / cal.sigma_global[int(l)])
    return float(np.mean(errs)) if errs else np.nan
