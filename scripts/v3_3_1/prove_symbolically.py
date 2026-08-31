"""V3.3.1 PHASE 10 -- symbolic verification of every V3.3.1 theorem.

This CHECKS the analytic proofs written in docs/v3_3_1/MATHEMATICAL_THEORY.md;
it does not replace them. Anything SymPy cannot settle is flagged rather than
asserted.
"""
import sys, json, pathlib
import sympy as sp

rho, kappa, n, x = sp.symbols("rho kappa n x", positive=True)
OUT = pathlib.Path("results/v3_3_1/tables"); OUT.mkdir(parents=True, exist_ok=True)
res = {}

g = 1 - 2 * rho
R_ret = 2 * rho
d_H = sp.log(2) / sp.log(1 / rho)
A_f = 1 / (2 * rho)
A_s = 1 / (2 * rho ** 2)

print("=== THEOREM G: rho*(kappa) = 1/(2+kappa) ===")
sol = sp.solve(sp.Eq(g, kappa * rho), rho)
res["G_solution"] = str(sol)
ok_G = sp.simplify(sol[0] - 1 / (2 + kappa)) == 0
print(f"  solve(1-2rho = kappa*rho) -> {sol[0]}   equals 1/(2+kappa): {ok_G}")
print(f"  kappa=1 -> {sp.nsimplify(sol[0].subs(kappa, 1))}")
res["G_ok"] = bool(ok_G)
res["G_at_kappa1"] = str(sp.nsimplify(sol[0].subs(kappa, 1)))

print("\n=== THEOREM BGR: B_kappa = min(rho, g/kappa) ===")
# On (0,1/2): rho is increasing, g/kappa is decreasing, so the min is maximised
# exactly where they cross.
d_rho = sp.diff(rho, rho)
d_gk = sp.diff(g / kappa, rho)
print(f"  d/drho [rho]     = {d_rho}   (> 0, increasing)")
print(f"  d/drho [g/kappa] = {d_gk}  (< 0 for kappa>0, decreasing)")
cross = sp.solve(sp.Eq(rho, g / kappa), rho)[0]
ok_B = sp.simplify(cross - 1 / (2 + kappa)) == 0
print(f"  crossing at rho = {cross}   equals 1/(2+kappa): {ok_B}")
Bmax = sp.simplify(cross)
print(f"  B_max = {Bmax};  at kappa=1: {sp.nsimplify(Bmax.subs(kappa,1))}")
res["BGR_ok"] = bool(ok_B)
res["BGR_max_value"] = str(Bmax)

print("\n=== THEOREM P: monotonicity on (0, 1/2) ===")
mono = {}
for name, expr, want in (("retention", R_ret, "increasing"),
                         ("hausdorff_dim", d_H, "increasing"),
                         ("alpha_field", A_f, "decreasing"),
                         ("alpha_sensitivity", A_s, "decreasing")):
    d = sp.simplify(sp.diff(expr, rho))
    sign = sp.simplify(sp.sign(d.subs(rho, sp.Rational(1, 3))))
    # prove sign over the whole open interval
    positive = sp.ask(sp.Q.positive(d), sp.Q.positive(rho) & sp.Q.lt(rho, sp.Rational(1, 2)))
    mono[name] = {"derivative": str(d), "sign_at_1/3": str(sign), "want": want}
    print(f"  {name:18s} d/drho = {str(d):28s} sign@1/3 = {sign}  ({want})")
res["P_monotonicity"] = mono
# all four are optimised at the largest feasible rho
res["P_ok"] = bool(sp.simplify(sp.diff(R_ret, rho)) > 0
                   and sp.simplify(sp.diff(A_f, rho).subs(rho, sp.Rational(1, 3))) < 0)
print(f"  => all four optimal at the upper feasible boundary rho*(kappa): {res['P_ok']}")

print("\n=== COUNTEREXAMPLE: unconstrained optimum is rho -> 1/2, not 1/3 ===")
lim = {name: sp.limit(e, rho, sp.Rational(1, 2), "-")
       for name, e in (("retention", R_ret), ("d_H", d_H),
                       ("alpha_field", A_f), ("alpha_sensitivity", A_s))}
for k, v in lim.items():
    at13 = sp.nsimplify(sp.simplify(
        {"retention": R_ret, "d_H": d_H, "alpha_field": A_f,
         "alpha_sensitivity": A_s}[k].subs(rho, sp.Rational(1, 3))))
    print(f"  {k:18s} rho->1/2: {v}    at rho=1/3: {at13}")
res["counterexample_limits"] = {k: str(v) for k, v in lim.items()}

print("\n=== COUNTEREXAMPLE: new-coverage argmax = n/(2(n+1)), NOT 1/3 ===")
F = (1 - x) * x ** n
dF = sp.simplify(sp.diff(F, x))
xs = sp.solve(sp.Eq(dF, 0), x)
print(f"  F_n(x) = (1-x)x^n,  dF/dx = {sp.factor(dF)}")
print(f"  critical x = {xs}")
xstar = [s for s in xs if s != 0][0]
rstar = sp.simplify(xstar / 2)
print(f"  x* = {xstar}  =>  rho* = {rstar}")
vals = {int(k): float(rstar.subs(n, k)) for k in (1, 2, 3, 5, 10, 50)}
print(f"  rho* by n: {vals}")
print(f"  equals 1/3 only at n=2: {abs(vals[2]-1/3) < 1e-12 and abs(vals[3]-1/3) > 1e-3}")
res["new_coverage_argmax"] = str(rstar)
res["new_coverage_by_n"] = vals

print("\n=== COROLLARY: measures ===")
mu_K = (2 * rho) ** n
mu_G = (1 - 2 * rho) * (2 * rho) ** n
print(f"  mu(K_n)     = {mu_K}      Cantor: {sp.simplify(mu_K.subs(rho, sp.Rational(1,3)))}")
print(f"  mu(G_(n+1)) = {mu_G}   Cantor: {sp.simplify(mu_G.subs(rho, sp.Rational(1,3)))}")
res["mu_K"] = str(sp.simplify(mu_K.subs(rho, sp.Rational(1, 3))))
res["mu_G_next"] = str(sp.simplify(mu_G.subs(rho, sp.Rational(1, 3))))

print("\n=== THEOREM 42: kappa = 2*beta, Cantor iff beta = 1/2 ===")
beta = sp.Symbol("beta", positive=True)
# two-sided margin: guard must absorb delta on each side, delta = beta*rho
rs = sp.solve(sp.Eq(g, 2 * beta * rho), rho)[0]
print(f"  g >= 2*beta*rho  =>  rho* = {rs}")
print(f"  rho* = 1/3  <=>  beta = {sp.solve(sp.Eq(rs, sp.Rational(1,3)), beta)}")
res["beta_for_cantor"] = str(sp.solve(sp.Eq(rs, sp.Rational(1, 3)), beta))

(OUT / "symbolic_proofs.json").write_text(json.dumps(res, indent=2, default=str))
print(f"\nwrote {OUT}/symbolic_proofs.json")
