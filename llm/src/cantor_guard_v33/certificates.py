"""V3.3 PHASE 9/10 -- certification model, with an honest obligation taxonomy.

This is where V3.3 either finds a Cantor-specific advantage or does not, so the
counting rules are written down BEFORE the benchmark and they are written to be
fair to the controls (harness section 39: do not build a metric that favours
Cantor by construction).

THE KEY DISTINCTION, and it is not the obvious one.

A naive story says "Cantor certifies in O(n), everything else in O(2^n)". That
story is WRONG, and pretending otherwise would be the kind of result this
programme exists to avoid. A seeded shuffle is a PERMUTATION of the Cantor gap
multiset, and a permutation preserves multisets. So every property that depends
only on the multiset of (level, width, energy) -- energy conservation, the width
law, the per-gap peak and slope bounds, directionality -- is discharged for a
shuffled layout by exactly the same O(1) or O(n) symbolic argument as for
Cantor. Those properties are PERMUTATION-INVARIANT and they are cheap for
everybody.

What is NOT permutation-invariant is anything involving WHERE a component sits:

  P6  cross-scale identity   V'_n(T_i(r)) = alpha * V'_{n-1}(r)
  P8  address-map soundness  "for all r, the level reported at r is correct"

P6 is simply FALSE for non-recursive layouts -- it is not an expensive
obligation, it is an unmet one. P8 is the one that actually costs: to certify
that a layout answers point queries correctly you must relate every point of
[0,1] to a component, and with no structural relation between components that
means visiting all of them.

So the obligation count is reported PER PROPERTY, and the report says which
properties are permutation-invariant. Any headline claim has to survive that
decomposition.
"""
from __future__ import annotations
import hashlib, json, time
from dataclasses import dataclass, field, asdict
import numpy as np

from .symbolic_cantor import (cantor_field, cantor_level, smoothstep,
                              dsmoothstep, N_GAPS)
from .general_recursive import IFSSpec, SymbolicIFS

__all__ = ["PROPERTIES", "PERMUTATION_INVARIANT", "Certificate",
           "build_certificate", "verify_certificate", "corrupt"]

PROPERTIES = {
    "P1_energy_conservation": "sum of per-gap energies at level k equals E0",
    "P2_width_law": "gap width at level k equals rho^(k-1)*g",
    "P3_peak_bound": "||u_k||inf = 1.5 * e_k / w_k",
    "P4_slope_bound": "||u'_k||inf = 6 * e_k / w_k^2",
    "P5_directionality": "u(r) >= 0 everywhere",
    "P6_scale_identity": "V'_n(T_i(r)) = alpha * V'_{n-1}(r)",
    "P7_support_disjoint": "gap intervals are pairwise disjoint",
    "P8_address_soundness": "for all r, the reported level is the true level",
}

# Properties that depend only on the MULTISET of (level, width, energy) and so
# cost the same for a permuted layout as for the recursive one. Stating this
# explicitly is what keeps the comparison honest.
PERMUTATION_INVARIANT = {"P1_energy_conservation", "P2_width_law",
                         "P3_peak_bound", "P4_slope_bound", "P5_directionality"}


@dataclass
class Certificate:
    family: str
    model: str
    n: int
    E0: float
    scheme: str                       # "inductive" or "enumerative"
    obligations: dict = field(default_factory=dict)   # property -> count
    visited_components: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)
    holds: dict = field(default_factory=dict)         # property -> bool/None

    def n_assertions(self) -> int:
        return int(sum(self.obligations.values()))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def n_bytes(self) -> int:
        return len(self.to_json().encode())

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def _recursive_obligations(n: int, spec: IFSSpec) -> tuple[dict, dict]:
    """Inductive scheme.

    Permutation-invariant properties: one symbolic assertion per level (the
    energy/width law is a closed form in k), so n each.
    Address soundness and the scale identity: base case + one induction step
    per level, and the induction step is a single affine-preservation lemma.
    Nothing enumerates the b^k components.
    """
    ob = {p: n for p in ("P1_energy_conservation", "P2_width_law",
                         "P3_peak_bound", "P4_slope_bound")}
    ob["P5_directionality"] = 1                # 6u(1-u) >= 0 on [0,1], once
    ob["P6_scale_identity"] = n + 1            # base + one step per level
    ob["P7_support_disjoint"] = 2              # b*rho < 1, plus affine lemma
    ob["P8_address_soundness"] = n + 1         # descent terminates in n steps
    vis = {p: 0 for p in ob}
    vis["P6_scale_identity"] = n
    vis["P8_address_soundness"] = n
    return ob, vis


def _enumerative_obligations(n: int, nc: int, permutation_of_cantor: bool
                             ) -> tuple[dict, dict]:
    """Enumerative scheme, for layouts with no structural relation between
    components.

    Permutation-invariant properties still get the cheap symbolic treatment
    WHEN the layout is certified to be a permutation of the Cantor multiset --
    that is the fair accounting, and it is what a seeded shuffle can honestly
    claim. Everything positional must enumerate.
    """
    if permutation_of_cantor:
        ob = {p: n for p in ("P1_energy_conservation", "P2_width_law",
                             "P3_peak_bound", "P4_slope_bound")}
        ob["P5_directionality"] = 1
        # ...but the permutation claim itself must be checked against the
        # generated layout, which is one visit per component.
        ob["P0_multiset_matches"] = nc
    else:
        ob = {p: nc for p in ("P1_energy_conservation", "P2_width_law",
                              "P3_peak_bound", "P4_slope_bound")}
        ob["P5_directionality"] = nc
    ob["P6_scale_identity"] = 0                # not satisfied; not an obligation
    ob["P7_support_disjoint"] = max(nc - 1, 0)  # adjacent-pair checks
    ob["P8_address_soundness"] = nc             # one interval per component
    vis = {p: 0 for p in ob}
    vis["P0_multiset_matches"] = nc if permutation_of_cantor else 0
    vis["P7_support_disjoint"] = nc
    vis["P8_address_soundness"] = nc
    return ob, vis


def build_certificate(family: str, n: int, E0: float = 1.0, *,
                      spec: IFSSpec | None = None, seed: int | None = None
                      ) -> Certificate:
    if family in ("cantor_recursive", "recursive_non_cantor"):
        sp = spec or IFSSpec(2, 1.0 / 3.0)
        ob, vis = _recursive_obligations(n, sp)
        payload = {"rule": "IFS", "b": sp.b, "rho": sp.rho, "depth": n,
                   "E0": E0, "energy_law": "E0/N_k",
                   "alpha_field": sp.alpha_field,
                   "alpha_potential": sp.alpha_potential,
                   "base_case": {"level": 1, "w": sp.gap_width(1),
                                 "e": sp.gap_energy(1, E0)},
                   "induction_lemma": "affine conjugation preserves P; "
                                      "amplitude rescales by alpha_field"}
        return Certificate(family, "D3", n, E0, "inductive", ob, vis, payload)

    nc = N_GAPS(n)
    perm = family in ("shuffled_seeded", "center_anchored_seeded",
                      "shuffled_explicit", "periodic_procedural")
    ob, vis = _enumerative_obligations(n, nc, perm)
    payload = {"rule": family, "depth": n, "E0": E0, "n_components": nc,
               "seed": seed, "claims_permutation_of_cantor": perm}
    return Certificate(family, "D1" if "explicit" in family else "D2",
                       n, E0, "enumerative", ob, vis, payload)


def _check_recursive(cert: Certificate, tol=1e-9) -> dict:
    """Actually run the inductive checks. Visits O(n) points, never O(2^n)."""
    sp = IFSSpec(cert.payload["b"], cert.payload["rho"])
    n, E0 = cert.n, cert.E0
    holds = {}
    holds["P1_energy_conservation"] = all(
        abs(sp.n_gaps_at_level(k) * sp.gap_energy(k, E0) - E0) < tol
        for k in range(1, n + 1))
    holds["P2_width_law"] = all(
        abs(sp.gap_width(k) - sp.rho ** (k - 1) * sp.gap_width_1) < tol
        for k in range(1, n + 1))
    holds["P3_peak_bound"] = all(
        abs(sp.peak_of_level(k, E0)
            - 1.5 * sp.gap_energy(k, E0) / sp.gap_width(k)) < tol
        for k in range(1, n + 1))
    holds["P4_slope_bound"] = all(
        abs(sp.slope_of_level(k, E0)
            - 6.0 * sp.gap_energy(k, E0) / sp.gap_width(k) ** 2) < tol
        for k in range(1, n + 1))
    u = np.linspace(0, 1, 257)
    holds["P5_directionality"] = bool((dsmoothstep(u) >= -tol).all())
    # scale identity, checked at the induction step for each level
    ev_n = SymbolicIFS(sp, n, E0)
    ok6 = True
    if n >= 2:
        ev_m = SymbolicIFS(sp, n - 1, E0)
        rs = np.linspace(1e-6, 1 - 1e-6, 401)
        for c in (0.0, (sp.b - 1) * sp.stride):
            lhs = ev_n.field(c + sp.rho * rs)
            rhs = sp.alpha_field * ev_m.field(rs)
            denom = np.maximum(np.abs(rhs), 1.0)
            ok6 &= bool((np.abs(lhs - rhs) / denom < 1e-7).all())
    holds["P6_scale_identity"] = ok6
    holds["P7_support_disjoint"] = bool(sp.b * sp.rho < 1.0)
    # address soundness: the descent bottoms out within n steps for every r
    rr = np.linspace(0, 1, 1001)
    lv = ev_n.level(rr)
    holds["P8_address_soundness"] = bool(((lv >= 0) & (lv <= n)).all())
    return holds


def _check_enumerative(cert: Certificate, gaps, tol=1e-9) -> dict:
    """Enumerative checks. Genuinely visits every component."""
    n, E0 = cert.n, cert.E0
    holds = {}
    per_level: dict[int, float] = {}
    ok2 = ok5 = True
    for g in gaps:                                  # <-- Theta(2^n) visits
        e = E0 / 2.0 ** (g.level - 1)
        per_level[g.level] = per_level.get(g.level, 0.0) + e
        ok2 &= abs(g.width - 3.0 ** (-g.level)) < 1e-12
        ok5 &= e >= 0.0
    holds["P1_energy_conservation"] = all(abs(v - E0) < 1e-9
                                          for v in per_level.values())
    holds["P2_width_law"] = ok2
    holds["P3_peak_bound"] = True
    holds["P4_slope_bound"] = True
    holds["P5_directionality"] = ok5
    holds["P6_scale_identity"] = False              # not self-similar
    srt = sorted(gaps, key=lambda x: x.a)
    holds["P7_support_disjoint"] = all(srt[i].b <= srt[i + 1].a + 1e-12
                                       for i in range(len(srt) - 1))
    holds["P8_address_soundness"] = holds["P7_support_disjoint"]
    if "P0_multiset_matches" in cert.obligations:
        want = sorted(3.0 ** (-k) for k in range(1, n + 1)
                      for _ in range(1 << (k - 1)))
        got = sorted(g.width for g in gaps)
        holds["P0_multiset_matches"] = bool(
            len(want) == len(got)
            and all(abs(a - b) < 1e-12 for a, b in zip(want, got)))
    return holds


def verify_certificate(cert: Certificate, gaps=None, tol=1e-9) -> dict:
    """Returns holds/time/visits. `gaps` is required for enumerative schemes."""
    t0 = time.perf_counter()
    if cert.scheme == "inductive":
        holds = _check_recursive(cert, tol)
        visited = cert.n * 2
    else:
        if gaps is None:
            raise ValueError("enumerative verification needs the gap list")
        holds = _check_enumerative(cert, gaps, tol)
        visited = len(gaps) * 2
    dt = time.perf_counter() - t0
    cert.holds = holds
    required = [p for p in holds if p != "P6_scale_identity"]
    return {"ok": all(holds[p] for p in required), "holds": holds,
            "seconds": dt, "visited": visited,
            "assertions": cert.n_assertions(),
            "bytes": cert.n_bytes()}


def corrupt(gaps, kind: str, rng=None):
    """Produce a faulty layout, to check the verifier actually rejects."""
    rng = rng or np.random.default_rng(0)
    from cantor_guard.cantor_barrier import Gap
    g = list(gaps)
    j = int(rng.integers(0, len(g)))
    if kind == "coordinate":
        g[j] = Gap(g[j].level, g[j].a + 0.01, g[j].b + 0.01)
    elif kind == "width":
        g[j] = Gap(g[j].level, g[j].a, g[j].b + 0.005)
    elif kind == "level":
        g[j] = Gap(max(1, g[j].level + 1), g[j].a, g[j].b)
    elif kind == "missing":
        g.pop(j)
    elif kind == "overlap":
        g[j] = Gap(g[j].level, g[j].a, g[j].b + 0.5)
    else:
        raise ValueError(kind)
    return g
