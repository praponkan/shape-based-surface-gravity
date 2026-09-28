#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures 4 and 5 for Paper V, the two that carry the central argument.

Figure 4: recovered density and excess mass against the split plane, for
Itokawa, 67P and Arrokoth. The reversal between the first panel and the other
two is the observation that Section 5 has to explain.

Figure 5: the scaling slope of log(drho) against log(V_R) versus signal
strength, for the synthetic family, with the three real bodies overlaid. A
slope of -1 means the excess mass is held fixed as the plane moves.

Inputs: all_results.json, paperV_runs.json, synth_invariance.json
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 8.5,
    "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "axes.labelpad": 3,
    "legend.frameon": False,
    "figure.dpi": 200,
})

C_RHO = "#1f4e79"      # density
C_DM = "#b3541e"       # excess mass
C_GREY = "#8c8c8c"


def new_runs():
    """The rerun records: quadric decimation, corrected clipper."""
    data = load("paperV_final.json", ("/mnt/project/paperV_final.json",))
    return [r for r in data if "meshcache" in r.get("command", "")
            and "--decimator cluster" not in r.get("command", "")]


def sweep(runs, frag, model="head", mass=None, fracsplit=None):
    rows = []
    for r in runs:
        if frag not in r["shape_model"]:
            continue
        if mass is not None and abs(r["M_total_kg"] - mass) > 1e-3 * mass:
            continue
        if fracsplit is True and "--fracsplit" not in r["command"]:
            continue
        if fracsplit is False and "--fracsplit" in r["command"]:
            continue
        m = r["models"].get(model)
        if not m or m.get("status") != "ok":
            continue
        if not m["minimum"]["interior_minimum"]:
            continue
        rows.append((m["volume_fraction_pct"], m["minimum"]["rho_R"],
                     m["minimum"]["rho_rest"],
                     m["derived"]["excess_mass_pct_of_total"],
                     m["minimum"]["improvement_over_uniform_pct"],
                     m["x_lo_km"]))
    a = np.array(sorted(set(rows)), float)
    return {"frac": a[:, 0], "rho": a[:, 1], "rest": a[:, 2], "dM": a[:, 3],
            "impr": a[:, 4], "x": a[:, 5]}


def load(path, alts=()):
    for p in (path,) + tuple(alts):
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    sys.exit(f"not found: {path}")


def itokawa_sweep(allres):
    sw = allres["CORRECTED_paperV_plane_clipped_regions"]["itokawa"][
        "split_plane_sweep"]["points"]
    pts = sorted(sw, key=lambda p: p["x_split_km"])
    return {
        "x": np.array([p["x_split_km"] for p in pts]),
        "rho": np.array([p["rho_head"] for p in pts], float),
        "rest": np.array([p["rho_rest"] for p in pts], float),
        "dM": np.array([100 * p["excess_mass_kg"] / 3.58e10 for p in pts]),
        "frac": np.array([p["volume_fraction_pct"] for p in pts]),
        "allowed": np.array([p["rho_head"] <= 3540 for p in pts]),
        "impr": np.array([p["improvement_pct"] for p in pts]),
    }


def arrokoth_frac_sweep(path="arrokoth_runs.json",
                        alt="/mnt/project/arrokoth_runs.json"):
    """The fraction-based Arrokoth sweep, which spans V_R by 1.89.

    The earlier position-based sweep moved the plane 3.5 km but changed the
    region volume by only 5%, because Arrokoth's neck is long and thin. It
    could not constrain the scaling and is superseded.
    """
    data = load(path, (alt,))
    rows = []
    for r in data:
        if "--fracsplit" not in r["command"]:
            continue
        m = r["models"].get("head")
        if not m or m["status"] != "ok" or not m["minimum"]["interior_minimum"]:
            continue
        rows.append((m["volume_fraction_pct"], m["minimum"]["rho_R"],
                     m["minimum"]["rho_rest"],
                     m["derived"]["excess_mass_pct_of_total"],
                     m["minimum"]["improvement_over_uniform_pct"],
                     r["fracsplit_x_km"]))
    rows.sort()
    a = np.array(rows, float)
    return {"frac": a[:, 0], "rho": a[:, 1], "rest": a[:, 2], "dM": a[:, 3],
            "impr": a[:, 4], "x": a[:, 5]}


def runs_sweep(runs, frag, mass=None, model="head"):
    rows = []
    for r in runs:
        if frag not in r["shape_model"]:
            continue
        if mass is not None and abs(r["M_total_kg"] - mass) > 1e-3 * mass:
            continue
        m = r["models"].get(model)
        if not m or m["status"] != "ok" or not m["minimum"]["interior_minimum"]:
            continue
        rows.append((m["x_lo_km"], m["minimum"]["rho_R"],
                     m["minimum"]["rho_rest"],
                     m["derived"]["excess_mass_pct_of_total"],
                     m["volume_fraction_pct"],
                     m["minimum"]["improvement_over_uniform_pct"]))
    rows.sort()
    a = np.array(rows, float)
    return {"x": a[:, 0], "rho": a[:, 1], "rest": a[:, 2],
            "dM": a[:, 3], "frac": a[:, 4], "impr": a[:, 5]}


MIN_VR_SPAN = 1.35     # below this the log-log fit is not constrained


def slope(frac, rho, rest, require_span=True):
    """Slope of log|drho| against log V_R, or None if it is not measurable.

    Returns None when the density difference changes sign (no anomaly to hold
    fixed) or when the sweep spans too little region volume for the fit to
    mean anything.
    """
    dr = rho - rest
    if dr.min() * dr.max() <= 0:
        return None
    if require_span and frac.max() / frac.min() < MIN_VR_SPAN:
        return None
    return float(np.polyfit(np.log(frac), np.log(np.abs(dr)), 1)[0])


# ------------------------------------------------------------------ fig 4
def figure4(bodies, out):
    """Both quantities as deviations from their own mean, on one common scale.

    Panels with independent axes make every curve look equally steep. Plotting
    the relative deviation puts them on the same footing, so the reversal
    between Itokawa and the other two is visible rather than merely stated.
    The abscissa is the region volume fraction, the variable the two
    quantities actually trade against, which also exposes how little of that
    range the Arrokoth sweep covered.
    """
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True)
    for ax, (name, d, note) in zip(axes, bodies):
        f = d["frac"]
        rr = 100 * (d["rho"] - d["rho"].mean()) / np.mean(np.abs(d["rho"]))
        # normalise by the mean of |dM| so a sign-flipping series stays finite
        dd = 100 * (d["dM"] - d["dM"].mean()) / np.mean(np.abs(d["dM"]))
        ax.plot(f, rr, "o-", color=C_RHO, ms=3.4, lw=1.2, mfc="white",
                mew=1.0, zorder=3)
        ax.plot(f, dd, "s--", color=C_DM, ms=3.2, lw=1.2, mfc="white",
                mew=1.0, zorder=3)
        ax.axhline(0, color=C_GREY, lw=0.6, ls=":", zorder=1)

        span = f.max() / f.min()
        ax.set_xlabel("region volume fraction (%)")
        ax.set_title(f"{name}\n{note},  $V_R$ span {span:.2f}",
                     fontsize=8.6, pad=5, linespacing=1.45)

        sl = slope(f, d["rho"], d["rest"])
        hr_rho = 100 * (d["rho"].max() - d["rho"].min()) / 2 / d["rho"].mean()
        crosses = d["dM"].min() * d["dM"].max() < 0
        hr_dm = (100 * (d["dM"].max() - d["dM"].min()) / 2
                 / abs(d["dM"].mean())) if not crosses else None
        if sl is not None:
            sl_txt = f"slope {sl:+.2f}"
        elif span < MIN_VR_SPAN:
            sl_txt = "slope not measurable"
        else:
            sl_txt = "$\\Delta\\rho$ changes sign"
        steadier = ("$\\Delta M$ steadier" if (hr_dm is not None
                                                and hr_dm < hr_rho)
                    else "$\\rho_R$ steadier")
        txt = (f"$\\rho_R$   $\\pm${hr_rho:.0f}%\n"
               + (f"$\\Delta M$   $\\pm${hr_dm:.0f}%" if hr_dm
                  else "$\\Delta M$   sign flip")
               + f"\n{sl_txt}\n{steadier}")
        ax.text(0.045, 0.965, txt, transform=ax.transAxes, va="top",
                ha="left", fontsize=7.1, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=C_GREY,
                          lw=0.5, alpha=0.95), zorder=5)

    axes[0].set_ylabel("deviation from the sweep mean\n"
                       "(% of the mean magnitude)")
    axes[0].set_yscale("symlog", linthresh=20, linscale=0.7)
    axes[0].set_ylim(-400, 400)
    axes[0].set_yticks([-300, -100, -20, 0, 20, 100, 300])
    axes[0].set_yticklabels(["-300", "-100", "-20", "0", "20", "100", "300"])
    axes[0].axhspan(-20, 20, color=C_GREY, alpha=0.07, lw=0)
    for a in axes[1:]:
        a.axhspan(-20, 20, color=C_GREY, alpha=0.07, lw=0)
    handles = [Line2D([], [], color=C_RHO, marker="o", ms=3.4, lw=1.2,
                      mfc="white", label="recovered density $\\rho_R$"),
               Line2D([], [], color=C_DM, marker="s", ms=3.2, lw=1.2,
                      ls="--", mfc="white", label="excess mass $\\Delta M$")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03), fontsize=8)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.70, bottom=0.17,
                        wspace=0.13)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------ fig 5
def figure5(syn, reals, no_anomaly, excluded, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    Y_NONE_S = -3.72
    withslope = [s for s in syn if s.get("scaling_slope") is not None]
    noslope = [s for s in syn if "scaling_slope" in s
               and s["scaling_slope"] is None]

    x = np.array([s["improvement_max_pct"] for s in withslope])
    y = np.array([s["scaling_slope"] for s in withslope])
    ax.plot(x, y, "o", color=C_RHO, ms=5.5, mfc="white", mew=1.3,
            label="synthetic, anomaly present", zorder=3)

    if noslope:
        xn = np.array([s["improvement_max_pct"] for s in noslope])
        ax.plot(xn, np.full_like(xn, Y_NONE_S), "v", color=C_GREY, ms=5,
                mfc=C_GREY, alpha=0.75, clip_on=False, zorder=3,
                label="synthetic, no anomaly ($\\Delta\\rho$ changes sign)")

    Y_NONE = -3.72
    for nm, impr, sl in reals:
        ax.plot(impr, sl, "*", ms=13, color=C_DM, mec="black", mew=0.5,
                zorder=4)
        dy = 7 if sl < -2.5 else -2
        ax.annotate(nm, (impr, sl), textcoords="offset points",
                    xytext=(8, dy), fontsize=7.8, color="black")
    for nm, impr in no_anomaly:
        ax.plot(impr, Y_NONE, "*", ms=13, color=C_DM, mec="black", mew=0.5,
                clip_on=False, zorder=4)
        ax.annotate(nm, (impr, Y_NONE), textcoords="offset points",
                    xytext=(8, 1), fontsize=7.8, color="black")

    ax.axhline(-1, color=C_DM, lw=1.0, ls="--", zorder=1)
    ax.text(0.985, -1.0, "  $\\Delta M$ held fixed", color=C_DM, fontsize=7.4,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.axhline(0, color=C_GREY, lw=0.8, ls=":", zorder=1)
    ax.text(0.985, 0.0, "  $\\Delta\\rho$ held fixed", color=C_GREY,
            fontsize=7.4, va="bottom", ha="right",
            transform=ax.get_yaxis_transform())

    ax.axvspan(0.9, 5.9, color=C_GREY, alpha=0.13, lw=0, zorder=0)
    ax.text(2.3, 0.28, "Paper IV noise band", fontsize=6.8, color=C_GREY,
            ha="center", va="center")

    if excluded:
        note = "; ".join(f"{n} omitted, $V_R$ span {s:.2f}" for n, s in excluded)
        ax.text(0.5, 1.02, note, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.8, color=C_GREY)
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.03, 0.40, f"correlation over\nthe synthetic set  {r:+.2f}",
            transform=ax.transAxes, ha="left", va="center", fontsize=7.2,
            color=C_RHO, linespacing=1.35)

    ax.set_xscale("log")
    ax.set_xlim(0.15, 70)
    ax.set_ylim(-3.95, 0.45)
    ax.set_xlabel("signal strength: dispersion improvement over uniform (%)")
    ax.set_ylabel("scaling slope  $d\\log\\Delta\\rho \\, / \\, d\\log V_R$")
    ax.text(0.015, Y_NONE_S, "no anomaly  ", color=C_GREY, fontsize=7.2,
            va="center", ha="left", transform=ax.get_yaxis_transform())
    ax.legend(loc="upper left", fontsize=7.2, bbox_to_anchor=(0.005, 0.80))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    allres = load("all_results.json", ("/mnt/project/all_results.json",))
    syn = load("synth_final.json", ("/mnt/project/synth_final.json",))
    NR = new_runs()

    it = itokawa_sweep(allres)
    keep = it["allowed"]
    it_a = {k: (v[keep] if isinstance(v, np.ndarray) else v)
            for k, v in it.items()}
    cg = sweep(NR, "cg_mspcd")
    ar = sweep(NR, "arrokoth", mass=1.6495e15, fracsplit=True)

    print("figure 4 inputs:")
    for nm, d in (("Itokawa", it_a), ("67P", cg), ("Arrokoth", ar)):
        sl = slope(d["frac"], d["rho"], d["rest"])
        print(f"  {nm:<9} n={len(d['x'])}  impr {d['impr'].min():.2f}-"
              f"{d['impr'].max():.2f}%  slope "
              f"{'none' if sl is None else f'{sl:+.2f}'}")

    figure4([("Itokawa", it_a,
              f"signal {it_a['impr'].min():.1f}$-${it_a['impr'].max():.1f}%"),
             ("67P", cg,
              f"signal {cg['impr'].min():.2f}$-${cg['impr'].max():.2f}%"),
             ("Arrokoth", ar, f"signal {ar['impr'].min():.2f}$-$"
                              f"{ar['impr'].max():.2f}%")],
            "fig4_rho_vs_dM.pdf")

    reals, no_anomaly, excluded = [], [], []
    for nm, d in (("Itokawa", it_a), ("67P", cg), ("Arrokoth", ar)):
        span = d["frac"].max() / d["frac"].min()
        dr = d["rho"] - d["rest"]
        sl = slope(d["frac"], d["rho"], d["rest"])
        if sl is not None:
            reals.append((nm, float(d["impr"].max()), sl))
        elif dr.min() * dr.max() <= 0:
            no_anomaly.append((nm, float(d["impr"].max())))
            print(f"  {nm}: no slope, the density difference changes sign "
                  f"(V_R span {span:.2f}) -> plotted on the no-anomaly row")
        else:
            excluded.append((nm, span))
            print(f"  {nm}: omitted, V_R span {span:.2f} below "
                  f"{MIN_VR_SPAN}")
    figure5(syn, reals, no_anomaly, excluded, "fig5_slope_vs_signal.pdf")


if __name__ == "__main__":
    main()
