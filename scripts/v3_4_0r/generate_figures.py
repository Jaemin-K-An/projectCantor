"""Phase 22 -- figures, each drawn from a stored table."""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "llm/src"))
from cantor_guard_v340.cantor_geometry import epsilon_h  # noqa: E402

from _common import CONFIG, RESULTS, V340, read_json  # noqa: E402

FIG = ROOT / "figures/v3_4_0r"
CANTOR, BASE, LIN = "#c0392b", "#7f8c8d", "#2980b9"


def _maybe(name, root=None):
    path = (root or RESULTS) / "tables" / name
    return read_json(path) if path.exists() else None


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print(f"   {name}")


def r01_historical_sensor():
    d = _maybe("sensor_vs_old_projection.json", V340)
    t = _maybe("sensor_transfer.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["old $z_v$\n(HarmfulQA)", "new $d_w$\n(HarmfulQA)"]
    values = [d["old_projection"]["auroc"], d["new_sensor"]["auroc"]]
    colors = [BASE, CANTOR]
    if t:
        labels.append("new $d_w$\n(LLM-LAT, frozen)")
        values.append(t["auroc"])
        colors.append("#27ae60")
    ax.bar(labels, values, color=colors)
    if t:
        lo, hi = t["auroc_ci95"]
        ax.errorbar([2], [t["auroc"]], yerr=[[t["auroc"] - lo], [hi - t["auroc"]]],
                    fmt="none", ecolor="k", capsize=6)
    ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUROC"); ax.grid(alpha=.3, axis="y")
    ax.set_title("V3.4.0 sensor result (historical) and its transfer to a new population")
    _save(fig, "R01_sensor_and_transfer.png")


def r03_budget_repair():
    old = _maybe("final_budget_audit.json", V340)
    cal = _maybe("budget_calibration.json")
    new = _maybe("final_budget_audit.json")
    if not (old and cal):
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    arms = [k for k in old["per_rho"]]
    x = np.arange(len(arms))
    ax.bar(x - 0.2, [old["per_rho"][a]["q_rms"] for a in arms], 0.4,
           label="V3.4.0 (clean-calibrated)", color=BASE)
    ax.axhline(old["target_q_rms"], color=BASE, ls="--", lw=1, label="V3.4.0 target 0.030")
    if new:
        vals = [new["per_arm"].get(a, {}).get("q_rms", np.nan) for a in arms]
        ax.bar(x + 0.2, vals, 0.4, label="V3.4.0R (attack-calibrated)", color=CANTOR)
        ax.axhline(new["target_q_rms"], color=CANTOR, ls="--", lw=1, label="V3.4.0R target 0.025")
    ax.set_xticks(x); ax.set_xticklabels(arms)
    ax.set_ylabel(r"realised $q_{rms}$"); ax.set_xlabel(r"$\rho$")
    ax.set_title("budget repair: calibrating on attacked states")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    _save(fig, "R03_budget_repair.png")


def r04_final_budget():
    audit = _maybe("final_budget_audit.json")
    if not audit:
        return
    arms = list(audit["per_arm"])
    vals = [audit["per_arm"][a]["q_rms"] for a in arms]
    target = audit["target_q_rms"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(arms, vals, color=[CANTOR if a == "1/3" else (LIN if a == "LINEAR" else BASE) for a in arms])
    ax.axhline(target, color="k", lw=1.5, label="target")
    ax.axhspan(target * 0.97, target * 1.03, color="#f1c40f", alpha=.25, label="±3% band")
    ax.set_ylabel(r"$q_{rms}$ on D_final_r"); ax.legend(); ax.grid(alpha=.3, axis="y")
    ax.set_title(f"final budget audit ({audit['verdict']})")
    _save(fig, "R04_final_budget.png")


def r06_07_controller_effect():
    eff = _maybe("controller_effect.json")
    if not eff:
        return
    frame = pd.read_csv(RESULTS / "raw" / "final_D_final_r_harmful.csv")
    col = eff["endpoint"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, family in zip(axes, ("ATTACK_W", "ATTACK_V")):
        block = frame[frame.family == family]
        for arm, group in block.groupby("arm"):
            curve = group.groupby("epsilon")[col].mean()
            style = dict(color=CANTOR, lw=2.4, zorder=5, label="Cantor 1/3") if arm == "1/3" else (
                dict(color="k", lw=2.4, ls="--", zorder=4, label="attack only") if arm == "ATTACK_ONLY" else (
                    dict(color=LIN, lw=2.0, zorder=3, label="linear") if arm == "LINEAR" else
                    dict(color=BASE, lw=0.9, alpha=.7)))
            ax.plot(curve.index, curve.values, "o-", markersize=3, **style)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel(r"attack magnitude $\varepsilon$"); ax.set_title(family); ax.grid(alpha=.3)
    axes[0].set_ylabel("safe rate"); axes[0].legend(fontsize=8)
    fig.suptitle("does the controller help? (dashed black = no controller)")
    _save(fig, "R06_controller_vs_no_controller.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (family, row) in zip(axes, eff["by_family"].items()):
        cs = row["max_t"]["contrasts"]
        for i, c in enumerate(cs):
            ax.plot([c["simultaneous_lo"], c["simultaneous_hi"]], [i, i], color="#2c3e50", lw=2)
            ax.plot([c["mean_difference"]], [i], "o", color=CANTOR, markersize=8)
        ax.axvline(0, color="k", lw=1)
        ax.axvspan(-eff["sesoi"], eff["sesoi"], color="#f1c40f", alpha=.18)
        ax.set_yticks(range(len(cs)))
        ax.set_yticklabels([f"{c['arm']} − {c['reference']}" for c in cs], fontsize=8)
        ax.set_xlabel("AUC difference"); ax.set_title(family); ax.grid(alpha=.3, axis="x")
    fig.suptitle("controller efficacy, max-T simultaneous intervals")
    _save(fig, "R08_efficacy_intervals.png")


def r09_10_rho_family():
    rho = _maybe("rho_family.json")
    if not rho:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (family, row) in zip(axes, rho["by_family"].items()):
        cs = row["max_t"]["contrasts"]
        for i, c in enumerate(cs):
            ax.plot([c["simultaneous_lo"], c["simultaneous_hi"]], [i, i], color="#2c3e50", lw=2)
            ax.plot([c["mean_difference"]], [i], "o", color=CANTOR, markersize=8)
        ax.axvline(0, color="k", lw=1)
        ax.axvspan(-rho["sesoi"], rho["sesoi"], color="#f1c40f", alpha=.18, label="SESOI")
        ax.set_yticks(range(len(cs)))
        ax.set_yticklabels([f"1/3 − {c['reference']}" for c in cs])
        ax.set_xlabel("AUC difference"); ax.set_title(family); ax.grid(alpha=.3, axis="x")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"rho family ({rho['cantor_verdict']})")
    _save(fig, "R10_rho_intervals.png")


def r11_certificate():
    cert = _maybe("certificate_validation.json")
    if not cert:
        return
    W = cert["W"]
    grid = np.linspace(0.001, 0.499, 3000)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grid, epsilon_h(grid, W), color="#2c3e50", lw=2)
    ax.axvline(1 / 3, color=CANTOR, ls="--", lw=2, label=r"$\rho=1/3$")
    for key, row in cert["per_rho"].items():
        ax.plot([row["rho"]], [row["epsilon_cert"]], "o",
                color=CANTOR if key == "1/3" else BASE, markersize=8, zorder=5)
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$\varepsilon_h(\rho)=2W\rho^2(1-2\rho)$")
    ax.set_title(f"structural policy-separation optimum ({cert['total_violations']} violations)")
    ax.legend(); ax.grid(alpha=.3)
    _save(fig, "R11_certificate.png")


def r12_survival():
    surv = _maybe("failure_survival.json")
    if not surv:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, family in zip(axes, ("ATTACK_W", "ATTACK_V")):
        for key, row in surv["by_arm_and_family"].items():
            if row["family"] != family or not row.get("curve"):
                continue
            arm = row["arm"]
            xs = [0] + [c["epsilon"] for c in row["curve"]]
            ys = [1.0] + [c["survival"] for c in row["curve"]]
            style = dict(color=CANTOR, lw=2.4, label="Cantor 1/3") if arm == "1/3" else (
                dict(color="k", lw=2.4, ls="--", label="attack only") if arm == "ATTACK_ONLY" else (
                    dict(color=LIN, lw=2.0, label="linear") if arm == "LINEAR" else
                    dict(color=BASE, lw=0.9, alpha=.6)))
            ax.step(xs, ys, where="post", **style)
        ax.axhline(0.5, color="#e67e22", ls=":", lw=1.5)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel(r"$\varepsilon$"); ax.set_title(family); ax.grid(alpha=.3)
    axes[0].set_ylabel("event-free survival"); axes[0].set_ylim(0, 1.02); axes[0].legend(fontsize=8)
    fig.suptitle("censor-aware behavioural survival (median only if the curve crosses 0.5)")
    _save(fig, "R12_survival.png")


def r14_15_utility():
    util = _maybe("utility.json")
    if not util:
        return
    arms = list(util["per_arm"])
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(arms))
    inside = [util["per_arm"][a]["inside_window"]["false_refusal"] or 0 for a in arms]
    outside = [util["per_arm"][a]["outside_window"]["false_refusal"] or 0 for a in arms]
    ax.bar(x - 0.2, inside, 0.4, label="inside window", color=LIN)
    ax.bar(x + 0.2, outside, 0.4, label="outside window", color="#e67e22")
    ax.axhline(util["no_controller_false_refusal"], color="k", ls="--", lw=1.5, label="no controller")
    ax.set_xticks(x); ax.set_xticklabels(arms, rotation=45)
    ax.set_ylabel("false refusal on benign prompts")
    ax.set_title(f"benign utility, split by window ({util['verdict']}; "
                 f"{util['benign_outside_window_rate']:.0%} outside)")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    _save(fig, "R15_utility_by_window.png")


def r16_chain():
    v = _maybe("final_verdict.json") or {}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    steps = [("SENSOR", v.get("SENSOR")), ("CERTIFICATE", v.get("CERTIFICATE")),
             ("BUDGET", v.get("BUDGET")), ("CONTROLLER_EFFECT", v.get("CONTROLLER_EFFECT")),
             ("CANTOR_BEHAVIOR", v.get("CANTOR_BEHAVIOR")), ("SEMANTIC", v.get("SEMANTIC")),
             ("UTILITY", v.get("UTILITY"))]
    for i, (name, value) in enumerate(steps):
        bad = value and any(t in str(value) for t in ("BUD2", "CTRL2", "CTRL3", "CANTOR4",
                                                      "CANTOR5", "PROXY", "FAIL", "BLOCKED"))
        ax.add_patch(plt.Rectangle((0.03, 0.87 - i * 0.125), 0.94, 0.10,
                                   color="#fdebd0" if bad else "#d5f5e3", ec="#7f8c8d"))
        ax.text(0.06, 0.92 - i * 0.125, name, fontsize=10, weight="bold", va="center")
        ax.text(0.34, 0.92 - i * 0.125, str(value), fontsize=9, va="center", family="monospace")
    ax.text(0.5, 0.02, f"OVERALL   {v.get('OVERALL')}", ha="center", fontsize=13,
            weight="bold", family="monospace")
    ax.set_title("V3.4.0R evidence chain")
    _save(fig, "R16_evidence_chain.png")


def main() -> None:
    print(f"figures -> {FIG}")
    for fn in (r01_historical_sensor, r03_budget_repair, r04_final_budget,
               r06_07_controller_effect, r09_10_rho_family, r11_certificate,
               r12_survival, r14_15_utility, r16_chain):
        try:
            fn()
        except Exception as exc:  # pragma: no cover
            print(f"   [skip] {fn.__name__}: {type(exc).__name__} {exc}")


if __name__ == "__main__":
    main()
