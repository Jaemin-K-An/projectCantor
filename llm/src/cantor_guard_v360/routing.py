"""Recursive leaf/guard routing, with explicit boundary ownership.

Internal guard endpoints belong to guards, as in V3.5.1.  Distances to
alternative terminal regions are INFIMA, not necessarily attained minima.
Safe/outside/guard transitions are deliberately not terminal switches.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def margin(rho, depth=3):
    if not isinstance(depth, (int, np.integer)) or depth < 1:
        raise ValueError("depth must be a positive integer")
    r = np.asarray(rho, dtype=float)
    if np.any(~np.isfinite(r)) or np.any((r <= 0) | (r >= .5)):
        raise ValueError("rho must lie in (0,1/2)")
    return r ** (depth - 1) * (1 - 2 * r)


def derivative(rho, depth=3):
    margin(rho, depth)
    r = np.asarray(rho, dtype=float)
    return -2.0 + np.zeros_like(r) if depth == 1 else r ** (depth - 2) * (depth - 1 - 2 * depth * r)


def optimal_rho(depth):
    margin(.3, depth)
    # n=1 has supremum at rho -> 0+, not an optimizer in the open domain.
    return None if depth == 1 else (depth - 1) / (2 * depth)


@dataclass(frozen=True)
class Region:
    lo: float
    hi: float
    index: int
    address: str
    depth: int


@lru_cache(maxsize=128)
def recursive_partition(rho, depth=3):
    margin(rho, depth)
    leaves, guards = [], []

    def split(lo, hi, level, path):
        if level == depth:
            leaves.append(Region(lo, hi, len(leaves), path, level))
            return
        a, b = lo + rho * (hi-lo), hi - rho * (hi-lo)
        guards.append(Region(a, b, len(guards), path, level+1))
        split(lo, a, level+1, path+'0')
        split(b, hi, level+1, path+'1')

    split(0., 1., 0, '')
    return tuple(leaves), tuple(sorted(guards, key=lambda g:g.lo))


class Router:
    def __init__(self, rho=1/3, depth=3, W_R=1.0, *, ablation=None):
        if not np.isfinite(W_R) or W_R <= 0:
            raise ValueError("W_R must be finite and positive")
        self.rho, self.depth, self.W_R = float(rho), depth, float(W_R)
        leaves, guards = recursive_partition(self.rho, depth)
        n = 2**depth
        if ablation == 'NO_GUARD':
            leaves = tuple(Region(i/n, (i+1)/n, i, '', depth) for i in range(n))
            guards = ()
        elif ablation == 'NON_RECURSIVE_GUARD':
            # Same leaf count AND total retained measure as the middle-third
            # reference. Equal gaps attain the largest possible minimum gap
            # under these aggregate constraints, without recursive nesting.
            width = (1/3)**depth
            gap = (1-n*width)/(n-1)
            leaves = tuple(Region(i*(width+gap), i*(width+gap)+width, i, '', depth) for i in range(n))
            guards = tuple(Region(leaves[i].hi, leaves[i+1].lo, i, '', 0) for i in range(n-1))
        elif ablation is not None:
            raise ValueError("unknown ablation")
        self.leaves, self.guards, self.ablation = leaves, guards, ablation

    @property
    def certificate(self):
        return self.W_R * float(margin(self.rho, self.depth)) if self.ablation is None else self.W_R * min(
            b.lo-a.hi for a,b in zip(self.leaves[:-1], self.leaves[1:]))

    def classify_x(self, x):
        """Return route kind, leaf index, conservative action, guard index."""
        x = np.asarray(x, float)
        if np.any(~np.isfinite(x)):
            raise ValueError("routing requires finite coordinates")
        r = x / self.W_R
        kind = np.full(r.shape, 'outside', dtype='<U8')
        leaf = np.full(r.shape, -1, int)
        guard = np.full(r.shape, -1, int)
        action = np.ones(r.shape)
        for c in self.leaves:
            mask = (r >= c.lo) & (r <= c.hi)
            kind[mask], leaf[mask], action[mask] = 'leaf', c.index, (c.index+1)/len(self.leaves)
        for i,g in enumerate(self.guards):
            mask = (r >= g.lo) & (r <= g.hi)
            right = next(c.index for c in self.leaves if c.lo >= g.hi)
            kind[mask], leaf[mask], action[mask], guard[mask] = 'guard', -1, (right+1)/len(self.leaves), i
        safe = x <= 0
        kind[safe], leaf[safe], guard[safe], action[safe] = 'safe', -1, -1, 0.
        return kind, leaf, action, guard

    def distances(self, d):
        """Exact residual-L2 infima for natural terminal-source states.

        For terminal x=-d>0 and target leaf J, the exact distance is
        dist(x, closure(J)). It is attained in the limit along sensor normal.
        Entries for nonterminal sources are NaN; they are never called stable.
        """
        d = np.atleast_1d(np.asarray(d, float))
        x = np.maximum(0., -d)
        kind, leaf, action, guard = self.classify_x(x)
        n = len(d)
        leave = np.full(n, np.nan); nearest_g = np.full(n, -1, int)
        guard_distance = np.full(n, np.nan); switch = np.full(n, np.nan)
        alternative = np.full(n, -1, int); target = np.full(n, np.nan)
        terminal = kind == 'leaf'
        if self.guards:
            g_lo = np.array([g.lo for g in self.guards])*self.W_R
            g_hi = np.array([g.hi for g in self.guards])*self.W_R
            all_g = np.maximum(np.maximum(g_lo[None,:]-x[:,None], x[:,None]-g_hi[None,:]),0)
            nearest_g[:] = all_g.argmin(1)
            guard_distance[:] = all_g.min(1)
            # d>=0 is folded; distance to a positive-risk guard includes d.
            guard_distance[d >= 0] = d[d>=0] + g_lo.min()
        for c in self.leaves:
            mask = terminal & (leaf == c.index)
            if not np.any(mask):
                continue
            v = x[mask]
            leave[mask] = np.minimum(v-c.lo*self.W_R, c.hi*self.W_R-v)
            candidates = [a for a in self.leaves if a.index != c.index]
            closest = np.stack([np.clip(v,a.lo*self.W_R,a.hi*self.W_R) for a in candidates], axis=1)
            distances = np.abs(closest-v[:,None])
            which = distances.argmin(1)
            switch[mask] = distances.min(1)
            alternative[mask] = np.array([a.index for a in candidates])[which]
            target[mask] = closest[np.arange(len(v)),which]
        return dict(d=d, x=x, kind=kind, leaf=leaf, action=action, guard=guard,
                    D_leave=leave, D_guard=guard_distance, D_terminal_switch=switch,
                    nearest_guard=nearest_g, alternative=alternative, target_x=target,
                    certificate_slack=switch-self.certificate)


def project_to_x(h, sensor, target_x):
    h = np.asarray(h, float)
    target = np.asarray(target_x, float)
    return h + (-target - np.asarray(sensor.distance(h)))[...,None] * sensor.w_hat


def numerical_switch_search(h, router, sensor, *, iterations=52, interior_fraction=1e-8):
    """Projected sensor-normal search evaluated via the actual routing code.

    Numerical search brackets the first entry into the nearest target leaf.
    Independent of the exact distance formula: terminal boundaries select a
    target interior, while bisection tests the implemented route predicate.
    Orthogonal components cannot decrease the bound (checked in validation).
    """
    h = np.asarray(h, float)
    d = np.asarray(sensor.distance(h))
    x = -d
    kind, source, _, _ = router.classify_x(np.maximum(0,-d))
    if not np.all(kind == 'leaf'):
        raise ValueError("numerical switch search requires terminal sources")
    target_index = np.empty(len(h), int); target_x = np.empty(len(h))
    for i, (v, src) in enumerate(zip(x, source)):
        options=[]
        for c in router.leaves:
            if c.index == src: continue
            lo,hi = c.lo*router.W_R,c.hi*router.W_R
            edge = lo if lo>v else hi
            options.append((abs(edge-v),c,edge))
        _,c,edge=min(options,key=lambda z:z[0])
        target_index[i]=c.index
        target_x[i]=edge + np.sign((c.lo+c.hi)*router.W_R/2-edge)*interior_fraction*(c.hi-c.lo)*router.W_R
    direction = -np.sign(target_x-x)[:,None] * sensor.w_hat
    # Measured slope from actual full residual perturbation, no distance oracle.
    slope = (direction @ sensor.w)/sensor.w_norm
    upper = np.abs(target_x-x); lower=np.zeros(len(h))
    for _ in range(iterations):
        mid=(lower+upper)/2
        k, idx, _, _=router.classify_x(np.maximum(0,-(d+mid*slope)))
        arrived=(k=='leaf')&(idx==target_index)
        upper=np.where(arrived,mid,upper); lower=np.where(arrived,lower,mid)
    # The boundary infimum can round to guard in a full vector operation.
    # Return both the numeric infimum estimate and a fixed interior witness.
    witness_h=h+np.abs(target_x-x)[:,None]*direction
    wk, wi, _, _=router.classify_x(np.maximum(0,-np.asarray(sensor.distance(witness_h))))
    return dict(estimate=(lower+upper)/2, witness_norm=np.linalg.norm(witness_h-h,axis=1),
                witness_success=(wk=='leaf')&(wi==target_index), target_index=target_index,
                lower=lower, upper=upper)
