"""Phase 23 -- figures.  Every panel is drawn from a stored table."""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "llm/src"))
from cantor_guard_v340.cantor_geometry import epsilon_h, partition  # noqa: E402

from _common import CONFIG, RESULTS, read_json  # noqa: E402

FIGURES = pathlib.Path(__file__).resolve().parents[2] / "figures/v3_4_0"
CANTOR = "#c0392b"


def _maybe(name):
    path = RESULTS / "tables" / name
    return read_json(path) if path.exists() else None


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print(f"   {name}")


def fig01_sensor_vs_old():
    d = _maybe("sensor_vs_old_projection.json")
    if not d:
        return
    keys = ["auroc", "pr_auc", "balanced_accuracy_at_zero"]
    labels = ["AUROC", "PR-AUC", "bal. acc."]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [d["old_projection"][k] for k in keys], 0.4, label=r"old  $z_v=\langle h,v\rangle$", color="#7f8c8d")
    ax.bar(x + 0.2, [d["new_sensor"][k] for k in keys], 0.4, label=r"new  $d_w$", color=CANTOR)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.5, 1.0)
    ax.set_title(f"sensor vs actuator projection (n={d['n']})"); ax.legend(); ax.grid(alpha=.3, axis="y")
    p = d["paired_bootstrap_auroc_new_minus_old"]
    ax2.errorbar([0], [p["auroc_difference_mean"]],
                 yerr=[[p["auroc_difference_mean"] - p["auroc_difference_ci95"][0]],
                       [p["auroc_difference_ci95"][1] - p["auroc_difference_mean"]]],
                 fmt="o", color=CANTOR, capsize=8, markersize=9)
    ax2.axhline(0, color="k", lw=1)
    ax2.set_xlim(-1, 1); ax2.set_xticks([])
    ax2.set_ylabel(r"$\Delta$AUROC (new $-$ old)")
    ax2.set_title(f"paired bootstrap\nangle(w,v) = {d['angle_w_v_deg']:.1f}°")
    ax2.grid(alpha=.3, axis="y")
    _save(fig, "F340-01_sensor_vs_old_projection.png")


def fig04_distance_distribution():
    path = RESULTS / "raw" / "clean_D_sensor_confirm.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    H = np.load(RESULTS / "cache" / "h_D_sensor_confirm.npy")
    fit = _maybe("sensor_confirm.json")
    w = np.load(RESULTS / "cache" / "sensor_w.npy")
    d = (H @ w + fit["b"]) / np.linalg.norm(w)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(d.min(), d.max(), 24)
    ax.hist(d[table.y_safe == 1], bins=bins, alpha=.65, label="safe / refusal", color="#2980b9")
    ax.hist(d[table.y_safe == 0], bins=bins, alpha=.65, label="compliance", color="#e67e22")
    ax.axvline(0, color=CANTOR, lw=2, label="learned boundary  $d=0$")
    ax.set_xlabel("signed sensor distance $d(h)$"); ax.set_ylabel("prompts")
    ax.set_title(f"held-out separation (AUROC {fit['confirm']['auroc']:.3f})")
    ax.legend(); ax.grid(alpha=.3)
    _save(fig, "F340-04_signed_distance_by_behaviour.png")


def fig05_stability():
    s = _maybe("sensor_stability.json")
    if not s:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["cosine_to_full_fit", "heldout_auroc", "decision_agreement_with_full_fit"]
    labels = ["cosine to full fit", "held-out AUROC", "decision agreement"]
    for i, key in enumerate(names):
        row = s[key]
        ax.errorbar([i], [row["mean"]],
                    yerr=[[row["mean"] - row["ci95"][0]], [row["ci95"][1] - row["mean"]]],
                    fmt="o", capsize=7, color=CANTOR, markersize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_ylim(0.8, 1.02); ax.grid(alpha=.3, axis="y")
    ax.set_title(f"hyperplane stability ({s['n_boot_effective']} training bootstraps)")
    _save(fig, "F340-05_hyperplane_stability.png")


def fig07_actuator():
    a = _maybe("actuator_validation.json")
    if not a:
        return
    rows = pd.DataFrame(a["by_dose"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rows.u, rows.y_safe, "o-", color=CANTOR, lw=2, markersize=8)
    ax.axvline(0, color="k", lw=1, ls=":")
    ax.set_xlabel(r"normalized dose $u$   ($\Delta h = u\,\|h\|\,v_{safe}$)")
    ax.set_ylabel("safe rate")
    ax.set_title(f"actuator revalidation  (Spearman {a['spearman_u_vs_safe']['rho']:+.3f}, "
                 f"p={a['spearman_u_vs_safe']['p']:.1e})")
    ax.grid(alpha=.3)
    _save(fig, "F340-07_actuator_response.png")


def fig09_partition_and_10_certificate():
    geom = _maybe("sensor_window_and_coupling.json")
    if not geom:
        return
    W = geom["W"]
    fig, ax = plt.subplots(figsize=(9, 4))
    for row, rho in enumerate(sorted(float(v["rho"]) for v in geom["certificates"].values())):
        leaves, guards = partition(rho, 3)
        for leaf in leaves:
            ax.add_patch(plt.Rectangle((leaf.lo, row - .35), leaf.hi - leaf.lo, .7,
                                       color="#2980b9", alpha=.75))
        for guard in guards:
            ax.add_patch(plt.Rectangle((guard.lo, row - .35), guard.hi - guard.lo, .7,
                                       color="#bdc3c7", alpha=.9))
    ax.axvline(0.5, color=CANTOR, lw=2, ls="--", label="$d=0$  ($r=1/2$)")
    labels = [("1/3" if abs(r - 1/3) < 1e-9 else f"{r:.2f}")
              for r in sorted(float(v["rho"]) for v in geom["certificates"].values())]
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlim(0, 1); ax.set_ylim(-.6, len(labels) - .4)
    ax.set_xlabel("risk coordinate $r$"); ax.set_ylabel(r"$\rho$")
    ax.set_title("depth-3 policy partition on the sensor coordinate (blue = leaf, grey = guard)")
    ax.legend(loc="upper right")
    _save(fig, "F340-09_cantor_partition.png")

    grid = np.linspace(0.001, 0.499, 4000)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, epsilon_h(grid, W), color="#2c3e50", lw=2)
    ax.axvline(1/3, color=CANTOR, ls="--", lw=2, label=r"$\rho=1/3$")
    for key, row in geom["certificates"].items():
        ax.plot([row["rho"]], [row["epsilon_h"]], "o",
                color=CANTOR if key == "1/3" else "#7f8c8d", markersize=8, zorder=5)
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\varepsilon_h(\rho)=2W\rho^2(1-2\rho)$")
    ax.set_title(f"certified residual radius, W={W:.4f}   "
                 rf"$\varepsilon_C=2W/27={geom['epsilon_cantor']:.4f}$")
    ax.legend(); ax.grid(alpha=.3)
    _save(fig, "F340-10_certificate_curve.png")


def fig11_12_generation():
    gen = _maybe("generation_analysis.json")
    if not gen:
        return
    frame = pd.read_csv(RESULTS / "raw" / "final_D_final_harmful.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, family in zip(axes, ("ATTACK_W", "ATTACK_V")):
        block = frame[frame.family.isin([family, "NONE"])]
        for key, group in block.groupby("rho_key"):
            curve = group.groupby("epsilon").y_safe.mean()
            ax.plot(curve.index, curve.values, "o-",
                    color=CANTOR if key == "1/3" else "#95a5a6",
                    lw=2.2 if key == "1/3" else 1.1,
                    zorder=5 if key == "1/3" else 1,
                    label=key if key == "1/3" else None, markersize=4)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel(r"attack magnitude $\varepsilon$  (residual $L_2$)")
        ax.set_title(family); ax.grid(alpha=.3)
    axes[0].set_ylabel("safe rate"); axes[0].legend()
    fig.suptitle("refusal robustness under matched-budget controllers (red = Cantor)")
    _save(fig, "F340-11_robustness_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (family, row) in zip(axes, gen["by_family"].items()):
        contrasts = row["max_t"]["contrasts"]
        y = np.arange(len(contrasts))
        for i, c in enumerate(contrasts):
            ax.plot([c["simultaneous_lo"], c["simultaneous_hi"]], [i, i], color="#2c3e50", lw=2)
            ax.plot([c["mean_difference"]], [i], "o", color=CANTOR, markersize=8)
        ax.axvline(0, color="k", lw=1)
        ax.axvspan(-gen["sesoi"], gen["sesoi"], color="#f1c40f", alpha=.18, label="SESOI")
        ax.set_yticks(y); ax.set_yticklabels([f"1/3 − {c['reference']}" for c in contrasts])
        ax.set_xlabel("AUC difference"); ax.set_title(family); ax.grid(alpha=.3, axis="x")
    axes[0].legend()
    fig.suptitle("max-T simultaneous intervals, preregistered Cantor contrasts")
    _save(fig, "F340-12_simultaneous_intervals.png")


def fig13_mechanism():
    m = _maybe("mechanism.json")
    if not m:
        return
    rows = pd.DataFrame(m["sensor_scores_by_family_and_rho"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, family in zip(axes, sorted(rows.family.unique())):
        block = rows[rows.family == family]
        x = np.arange(len(block))
        ax.bar(x - .25, block.mean_d_clean, .25, label="clean", color="#2980b9")
        ax.bar(x, block.mean_d_attacked, .25, label="attacked", color="#e74c3c")
        ax.bar(x + .25, block.mean_d_corrected, .25, label="corrected", color="#27ae60")
        ax.set_xticks(x); ax.set_xticklabels(block.rho_key, rotation=45)
        ax.axhline(0, color="k", lw=1); ax.set_title(family); ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("mean sensor distance $d$"); axes[0].legend()
    fig.suptitle("attacked → detected → corrected")
    _save(fig, "F340-13_sensor_restoration.png")


def fig17_utility():
    u = _maybe("utility.json")
    if not u:
        return
    rows = pd.DataFrame(u["per_arm"]).T.reset_index().rename(columns={"index": "arm"})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(rows.arm, rows.false_refusal.astype(float),
           color=[CANTOR if a == "1/3" else "#95a5a6" for a in rows.arm])
    ax.axhline(u["no_controller_false_refusal"], color="k", ls="--", lw=1.5, label="no controller")
    ax.set_ylabel("false refusal on benign prompts"); ax.set_xlabel(r"$\rho$")
    ax.set_title(f"benign utility ({u['verdict']})"); ax.legend(); ax.grid(alpha=.3, axis="y")
    _save(fig, "F340-17_benign_utility.png")


def fig18_evidence_chain():
    verdict = _maybe("final_verdict.json") or {}
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off")
    steps = [("SENSOR", verdict.get("SENSOR")), ("ACTUATOR", verdict.get("ACTUATOR")),
             ("COUPLING", verdict.get("COUPLING")), ("CERTIFICATE", verdict.get("CERTIFICATE")),
             ("SEMANTIC", verdict.get("SEMANTIC")), ("GENERATION", verdict.get("GENERATION")),
             ("UTILITY", verdict.get("UTILITY"))]
    for i, (name, value) in enumerate(steps):
        ok = value and not any(t in str(value) for t in ("NOT_", "FAIL", "PROXY", "SENS3", "ACT2", "COUP2"))
        ax.add_patch(plt.Rectangle((0.03, 0.88 - i * 0.125), 0.94, 0.10,
                                   color="#d5f5e3" if ok else "#fdebd0", ec="#7f8c8d"))
        ax.text(0.06, 0.93 - i * 0.125, name, fontsize=11, weight="bold", va="center")
        ax.text(0.30, 0.93 - i * 0.125, str(value), fontsize=10, va="center", family="monospace")
    ax.text(0.5, 0.02, f"OVERALL   {verdict.get('OVERALL')}", ha="center",
            fontsize=13, weight="bold", family="monospace")
    ax.set_title("V3.4.0 evidence chain: sensor → Cantor → actuator")
    _save(fig, "F340-18_evidence_chain.png")


def main() -> None:
    print(f"figures -> {FIGURES}")
    for fn in (fig01_sensor_vs_old, fig04_distance_distribution, fig05_stability,
               fig07_actuator, fig09_partition_and_10_certificate, fig11_12_generation,
               fig13_mechanism, fig17_utility, fig18_evidence_chain):
        try:
            fn()
        except Exception as exc:  # pragma: no cover
            print(f"   [skip] {fn.__name__}: {type(exc).__name__} {exc}")


if __name__ == "__main__":
    main()
