"""V3.3 PHASE 11-13/15 -- structural, symbolic-evaluation, scale-transfer and
certification benchmarks, over depth.

THEORY and MEASURED are reported side by side (harness section 28), and
symbolic vs materialised cost is never conflated (section 29).
"""
import sys, json, time, tracemalloc, pathlib, statistics
sys.path.insert(0, "llm/src")
import numpy as np

from cantor_guard.cantor_barrier import cantor_gap_list, BarrierLayout
from cantor_guard_v33.symbolic_cantor import (SymbolicCantor, cantor_field,
                                              N_GAPS)
from cantor_guard_v33.general_recursive import IFSSpec, CANTOR, SymbolicIFS
from cantor_guard_v33.complexity import describe, FAMILIES
from cantor_guard_v33.certificates import build_certificate, verify_certificate

OUT = pathlib.Path("results/v3_3/tables"); OUT.mkdir(parents=True, exist_ok=True)
NON_CANTOR = IFSSpec(2, 0.28, "recursive_non_cantor")
DEPTHS = list(range(1, 21))
REPS = 30
SEED = 20260831
E0 = 1.0

# Families whose point query is closed-form addressable (no materialisation).
CLOSED_FORM = {"cantor_recursive", "recursive_non_cantor", "periodic_procedural"}


def _kw(fam):
    k = {"seed": SEED}
    if fam == "recursive_non_cantor":
        k["ifs"] = NON_CANTOR
    return k


def bench_structure():
    rows, metrics = [], {}
    for n in DEPTHS:
        metrics[str(n)] = {}
        for fam in FAMILIES:
            d = describe(fam, n, **_kw(fam))
            m = {"M1_canonical_bits": d.canonical_bits,
                 "M3_point_query_words": d.storage_words_symbolic,
                 "materialised_words": d.materialised_words,
                 "serialized_bytes": d.serialized_bytes,
                 "gzip_bytes": d.gzip_bytes,
                 "ast_nodes": d.ast_nodes,
                 "explicit_params": d.explicit_params}
            metrics[str(n)][fam] = m
            rows.append({"n": n, "family": fam, "model": d.model,
                         "n_components": d.n_components,
                         "exact_scale_transfer": d.exact_scale_transfer, **m})
    return rows, metrics


def bench_certification(metrics):
    rows = []
    for n in DEPTHS:
        gaps = cantor_gap_list(n) if n <= 17 else None
        for fam in FAMILIES:
            kw = {}
            if fam == "recursive_non_cantor":
                kw["spec"] = NON_CANTOR
            elif "seed" in fam or "shuffled" in fam:
                kw["seed"] = SEED
            cert = build_certificate(fam, n, E0, **kw)
            need_gaps = cert.scheme == "enumerative"
            if need_gaps and gaps is None:
                # too large to materialise; record the theoretical obligation
                res = {"ok": None, "seconds": float("nan"),
                       "visited": 2 * N_GAPS(n),
                       "assertions": cert.n_assertions(),
                       "bytes": cert.n_bytes()}
            else:
                ts = []
                for _ in range(REPS if n <= 12 else 3):
                    r = verify_certificate(cert, gaps=gaps if need_gaps else None)
                    ts.append(r["seconds"])
                res = {"ok": r["ok"], "seconds": statistics.median(ts),
                       "p95": float(np.quantile(ts, .95)),
                       "visited": r["visited"], "assertions": r["assertions"],
                       "bytes": r["bytes"]}
            metrics[str(n)][fam]["M2_certificate_assertions"] = res["assertions"]
            metrics[str(n)][fam]["certificate_bytes"] = res["bytes"]
            metrics[str(n)][fam]["verification_seconds"] = res["seconds"]
            metrics[str(n)][fam]["visited_components"] = res["visited"]
            rows.append({"n": n, "family": fam, "scheme": cert.scheme, **res})
    return rows


def bench_symbolic_eval():
    """Point-query cost and peak memory: symbolic vs materialised."""
    rng = np.random.default_rng(SEED)
    rows = []
    for n in DEPTHS:
        q = rng.uniform(0, 1, 2000)
        tracemalloc.start()
        t0 = time.perf_counter()
        sym = SymbolicCantor(n, E0)
        vs = sym.field(q)
        t_sym = time.perf_counter() - t0
        _, peak_sym = tracemalloc.get_traced_memory(); tracemalloc.stop()

        t_exp = float("nan"); peak_exp = float("nan"); err = float("nan")
        if n <= 17:
            tracemalloc.start()
            t0 = time.perf_counter()
            L = BarrierLayout(cantor_gap_list(n), n, E0)
            ve = L.field(q)
            t_exp = time.perf_counter() - t0
            _, peak_exp = tracemalloc.get_traced_memory(); tracemalloc.stop()
            scale = max(1.0, float(np.abs(ve).max()))
            err = float(np.abs(ve - vs).max() / scale)
        rows.append({"n": n, "n_components": N_GAPS(n),
                     "symbolic_seconds": t_sym, "explicit_seconds": t_exp,
                     "symbolic_peak_bytes": peak_sym,
                     "explicit_peak_bytes": peak_exp,
                     "max_rel_error": err})
    return rows


def bench_scale_transfer():
    """Zero-shot extension: build C_{n+1} from the SAME rule, no refitting.

    E_scale = sup_r |u_{n+1}(T_i(r)) - alpha * u_n(r)| , relative.
    Reported for Cantor, the recursive non-Cantor family, and -- as the
    control -- the same affine maps applied to a shuffled layout, where no
    such identity is expected.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    r = np.linspace(1e-9, 1 - 1e-9, 4000)
    for n in range(2, 17):
        for name, spec in (("cantor_recursive", CANTOR),
                           ("recursive_non_cantor", NON_CANTOR)):
            hi, lo = SymbolicIFS(spec, n, E0), SymbolicIFS(spec, n - 1, E0)
            worst = 0.0
            for c in (0.0, (spec.b - 1) * spec.stride):
                lhs = hi.field(c + spec.rho * r)
                rhs = spec.alpha_field * lo.field(r)
                den = np.maximum(np.abs(rhs), 1.0)
                worst = max(worst, float((np.abs(lhs - rhs) / den).max()))
            rows.append({"n": n, "family": name, "E_scale_rel": worst,
                         "new_parameters": 0, "optimisation_calls": 0,
                         "zero_shot": True})
        if n <= 15:
            gaps = cantor_gap_list(n)
            order = rng.permutation(len(gaps))
            from cantor_guard.cantor_barrier import layout_from_order
            Ln = BarrierLayout(layout_from_order(gaps, order, n), n, E0)
            gm = cantor_gap_list(n - 1)
            om = rng.permutation(len(gm))
            Lm = BarrierLayout(layout_from_order(gm, om, n - 1), n - 1, E0)
            lhs = Ln.field(r / 3.0)
            rhs = 1.5 * Lm.field(r)
            den = np.maximum(np.abs(rhs), 1.0)
            rows.append({"n": n, "family": "shuffled_seeded",
                         "E_scale_rel": float((np.abs(lhs - rhs) / den).max()),
                         "new_parameters": N_GAPS(n) * 3,
                         "optimisation_calls": 0, "zero_shot": False})
    return rows


def bench_general_ifs():
    """(b, rho) trade-off map. NOT a search for a setting where Cantor wins --
    it locates Cantor inside the family."""
    rows = []
    for b in (2, 3, 4, 5):
        for rho in np.arange(0.05, 0.50, 0.01):
            if not (b * rho < 1.0):
                continue
            s = IFSSpec(int(b), float(rho))
            rows.append({"b": b, "rho": round(float(rho), 4),
                         "alpha_field": s.alpha_field,
                         "alpha_sensitivity": s.alpha_sensitivity,
                         "gap_width_1": s.gap_width_1,
                         "n_components_10": s.n_components(10),
                         "peak_k5": s.peak_of_level(5, E0),
                         "slope_k5": s.slope_of_level(5, E0),
                         "is_cantor": (b == 2 and abs(rho - 1 / 3) < 5e-3)})
    return rows


if __name__ == "__main__":
    import pandas as pd
    print("structure ...", flush=True)
    srows, metrics = bench_structure()
    print("certification ...", flush=True)
    crows = bench_certification(metrics)
    print("symbolic evaluation ...", flush=True)
    erows = bench_symbolic_eval()
    print("scale transfer ...", flush=True)
    trows = bench_scale_transfer()
    print("general IFS ...", flush=True)
    grows = bench_general_ifs()

    pd.DataFrame(srows).to_csv(OUT / "bench_structure.csv", index=False)
    pd.DataFrame(crows).to_csv(OUT / "bench_certification.csv", index=False)
    pd.DataFrame(erows).to_csv(OUT / "bench_symbolic_eval.csv", index=False)
    pd.DataFrame(trows).to_csv(OUT / "bench_scale_transfer.csv", index=False)
    pd.DataFrame(grows).to_csv(OUT / "bench_general_ifs.csv", index=False)

    summary = {"families": FAMILIES, "depths": DEPTHS,
               "deepest_n": DEPTHS[-1], "metrics": metrics,
               "reps": REPS, "seed": SEED,
               "non_cantor_ifs": {"b": NON_CANTOR.b, "rho": NON_CANTOR.rho}}
    (OUT / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/benchmark_summary.json")

    M = metrics[str(DEPTHS[-1])]
    print(f"\n--- primary metrics at n={DEPTHS[-1]} ---")
    print(f"{'family':26s} {'M1 bits':>9s} {'M2 assert':>11s} {'M3 words':>9s}")
    for f in FAMILIES:
        print(f"{f:26s} {M[f]['M1_canonical_bits']:9d} "
              f"{M[f]['M2_certificate_assertions']:11d} "
              f"{M[f]['M3_point_query_words']:9d}")
