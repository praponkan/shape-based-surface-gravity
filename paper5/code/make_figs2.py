#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures 1, 2, 3 and 7 for Paper V.

1  Cone versus plane-cut lobe on the unit sphere, with the closed-form values.
2  Dispersion curves at the reference plane for all five bodies, normalised so
   they can be compared: one deep well and four shallow ones.
3  Mesh convergence of the recovered density, Itokawa and Arrokoth.
7  Arrokoth contrast against the assumed bulk density, with the sign reversal.

Figures 4 and 5 are produced by make_figs.py.
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.size": 8.5, "axes.linewidth": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False, "figure.dpi": 200,
})

C_RHO, C_DM, C_GREY = "#1f4e79", "#b3541e", "#8c8c8c"
BODY_C = {"Itokawa": "#b3541e", "67P": "#1f4e79", "Arrokoth": "#2e7d32",
          "Eros": "#6a4c93", "Bennu": "#8c8c8c"}


def load(path, alts=()):
    for p in (path,) + tuple(alts):
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    sys.exit(f"not found: {path}")


# ------------------------------------------------------------------ fig 1
def figure1(out):
    R, XS = 1.0, 0.5
    h = R - XS
    V_cap = np.pi * h ** 2 * (3 * R - h) / 3
    x_cap = 3 * (2 * R - h) ** 2 / (4 * (3 * R - h))
    V_cone = (1 / 3) * R * (2 * np.pi * R * h)
    x_cone = 0.75 * (R - h / 2)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9))
    th = np.linspace(0, 2 * np.pi, 400)
    th_cap = np.linspace(-np.arccos(XS), np.arccos(XS), 200)

    for ax, kind in zip(axes, ("lobe", "cone")):
        ax.plot(np.cos(th), np.sin(th), color="black", lw=0.9, zorder=3)
        ax.axvline(XS, color=C_GREY, lw=0.8, ls="--", zorder=2)
        if kind == "lobe":
            pts = np.column_stack([np.cos(th_cap), np.sin(th_cap)])
            poly = np.vstack([pts, [[XS, pts[-1, 1]], [XS, pts[0, 1]]]])
            ax.add_patch(Polygon(poly, closed=True, fc=C_DM, alpha=0.30,
                                 ec=C_DM, lw=1.2, zorder=1))
            ax.plot(x_cap, 0, "o", color=C_DM, ms=5, zorder=4)
            title = "plane-cut lobe  (intended)"
            txt = (f"volume    {V_cap:.4f}\ncentroid $x$   {x_cap:.4f}")
        else:
            pts = np.column_stack([np.cos(th_cap), np.sin(th_cap)])
            poly = np.vstack([[[0.0, 0.0]], pts])
            ax.add_patch(Polygon(poly, closed=True, fc=C_RHO, alpha=0.30,
                                 ec=C_RHO, lw=1.2, zorder=1))
            ax.plot(0, 0, "o", color="black", ms=3.5, zorder=4)
            ax.annotate("apex at the\ninterior point $O$", (0, 0),
                        textcoords="offset points", xytext=(-6, -30),
                        fontsize=7, ha="center", color="black",
                        arrowprops=dict(arrowstyle="-", lw=0.6,
                                        color="black"))
            ax.plot(x_cone, 0, "o", color=C_RHO, ms=5, zorder=4)
            title = "cone anchored at $O$  (what was built)"
            txt = (f"volume    {V_cone:.4f}\ncentroid $x$   {x_cone:.4f}")
        ax.text(XS + 0.03, -1.16, "$x=0.5$", fontsize=7, color=C_GREY)
        ax.set_title(title, fontsize=8.6, pad=5)
        ax.text(-1.12, 1.02, txt, fontsize=7.6, va="top", ha="left",
                linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=C_GREY,
                          lw=0.5))
        ax.set_aspect("equal")
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])

    fig.text(0.5, -0.02,
             f"unit sphere cut at $x=0.5$:  the cone has "
             f"{V_cone / V_cap:.3f} times the volume and its centroid lies "
             f"{x_cap - x_cone:.4f}$R$ further in.\n"
             f"Both solids are watertight and both have exact volumes; only "
             f"the predicate test distinguishes them.",
             ha="center", va="top", fontsize=7.4, color="black",
             linespacing=1.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------ fig 2
def figure2(curves, out):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for nm, rho, disp, bulk, d_unif, impr in curves:
        ax.plot(np.array(rho) / bulk, np.array(disp) / d_unif,
                "-", lw=1.4, color=BODY_C[nm], label=f"{nm}  ({impr:.2f}%)")
        k = int(np.argmin(disp))
        ax.plot(rho[k] / bulk, disp[k] / d_unif, "o", ms=4.5,
                color=BODY_C[nm], mfc="white", mew=1.3, zorder=4)
    ax.axhline(1.0, color=C_GREY, lw=0.8, ls=":", zorder=1)
    ax.text(0.02, 1.005, "uniform density", fontsize=7, color=C_GREY,
            transform=ax.get_yaxis_transform(), va="bottom")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(0.78, 1.62)
    ax.set_xlabel("region density / bulk density")
    ax.set_ylabel("dispersion / dispersion of the uniform body")
    ax.legend(title="improvement at the minimum", fontsize=7.4,
              title_fontsize=7.4, loc="upper left", ncol=2,
              columnspacing=1.0, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------ fig 3
def figure3(series, out):
    """Recovered density against facet count, with the scan step shown.

    The scan step is the resolution of the answer, so a change smaller than
    one step is not a change at all. Drawing it as a band keeps the flat lines
    from being read as either more or less impressive than they are.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    for ax, (title, groups, ylab, step) in zip(axes, series):
        ref = None
        for lab, n, rho, style in groups:
            ax.plot(n, rho, style, ms=5.5, lw=1.2, mfc="white", mew=1.3,
                    label=lab)
            if ref is None:
                ref = rho[-1]
        for lab, n, rho, style in groups:
            ax.axhspan(rho[-1] - step, rho[-1] + step, color=C_GREY,
                       alpha=0.16, lw=0, zorder=0)
        ax.set_xlabel("facets used")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8.8, pad=5)
        allv = [v for _, _, r, _ in groups for v in r]
        m = 0.5 * (min(allv) + max(allv))
        half = max(0.5 * (max(allv) - min(allv)) * 1.9, 4.0 * step)
        ax.set_ylim(m - half, m + half)
        ax.legend(fontsize=7.2, loc="upper right")
        ax.text(0.03, 0.04, f"shaded: $\\pm$ one scan step ({step:.0f} "
                            f"kg m$^{{-3}}$)", transform=ax.transAxes,
                fontsize=6.9, color=C_GREY, va="bottom")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------ fig 7
def figure7(rho_assumed, contrast, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    x = np.array(rho_assumed, float)
    y = np.array(contrast, float)
    xs = np.linspace(200, 560, 300)
    c = np.polyfit(x, y, 2)
    ax.plot(xs, np.polyval(c, xs), "-", lw=1.2, color=C_GREY, zorder=1)
    ax.plot(x, y, "o", ms=6, color=BODY_C["Arrokoth"], mfc="white", mew=1.6,
            zorder=3)
    for a, b in zip(x, y):
        ax.annotate(f"{b:.3f}", (a, b), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.2)
    ax.axhline(1.0, color=C_DM, lw=1.0, ls="--", zorder=2)
    rts = [t.real for t in np.roots([c[0], c[1], c[2] - 1.0])
           if 150 < t.real < 700]
    if rts:
        xc = min(rts)
        ax.plot([xc], [1.0], "*", ms=13, color=C_DM, mec="black", mew=0.5,
                zorder=4)
        ax.annotate(f"sign reversal\nat {xc:.0f} kg m$^{{-3}}$", (xc, 1.0),
                    textcoords="offset points", xytext=(10, 14), fontsize=7.4,
                    color=C_DM, linespacing=1.3)
    ax.axvspan(200, 290, color=C_GREY, alpha=0.16, lw=0, zorder=0)
    ax.text(245, 0.905, "excluded by\nSpencer et al.\n(2020)", fontsize=6.9,
            color=C_GREY, ha="center", va="center", linespacing=1.3)
    ax.text(0.97, 0.94, "Wenu denser", transform=ax.transAxes, ha="right",
            fontsize=7.2, color=C_DM)
    ax.text(0.97, 0.06, "Wenu lighter", transform=ax.transAxes, ha="right",
            fontsize=7.2, color=C_DM)
    ax.set_xlim(200, 560)
    ax.set_xlabel("assumed bulk density (kg m$^{-3}$)")
    ax.set_ylabel("recovered contrast  $\\rho_{\\rm Wenu}/\\rho_{\\rm rest}$")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    allres = load("all_results.json", ("/mnt/project/all_results.json",))
    rec = load("paperV_final.json", ("/mnt/project/paperV_final.json",))
    runs = [r for r in rec if "meshcache" in r.get("command", "")
            and "--decimator cluster" not in r.get("command", "")]
    arro = runs
    root = allres["CORRECTED_paperV_plane_clipped_regions"]

    figure1("fig1_cone_vs_lobe.pdf")

    # ---- fig 2: reference dispersion curves ----------------------------
    curves = []
    it = root["itokawa"]["head_model"]
    curves.append(("Itokawa", [p["rho_R"] for p in it["scan"]],
                   [p["dispersion"] for p in it["scan"]],
                   root["itokawa"]["bulk_density_kg_m3"],
                   root["itokawa"]["uniform_reference_dispersion"],
                   it["minimum"]["improvement_over_uniform_pct"]))
    er = root["eros_negative_control"]["head_model_at_true_saddle"]
    curves.append(("Eros", [p["rho_R"] for p in er["scan"]],
                   [p["dispersion"] for p in er["scan"]],
                   root["eros_negative_control"]["configuration"][
                       "bulk_density_kg_m3"],
                   root["eros_negative_control"]["configuration"][
                       "uniform_reference_dispersion"],
                   er["minimum"]["improvement_over_uniform_pct"]))
    for nm, frag, src in (("67P", "cg_mspcd", runs), ("Bennu", "Bennu", runs)):
        for r in src:
            if frag not in r["shape_model"]:
                continue
            m = r["models"].get("head")
            if not m or m["status"] != "ok" or "scan" not in m:
                continue
            if nm == "67P" and abs(m["x_lo_km"] - 0.9517) > 0.02:
                continue
            curves.append((nm, [p["rho_R"] for p in m["scan"]],
                           [p["dispersion"] for p in m["scan"]],
                           r["bulk_density_kg_m3"],
                           r["uniform_reference_dispersion"],
                           m["minimum"]["improvement_over_uniform_pct"]))
            break
    # Arrokoth at its detected neck, from the original run, so that every
    # body in this figure is shown at its own reference plane
    for r in runs:
        if "arrokoth" not in r["shape_model"]:
            continue
        if abs(r["M_total_kg"] - 1.6495e15) > 1e12:
            continue
        if "--fracsplit" in r["command"]:
            continue
        m = r["models"].get("head")
        if not m or m["status"] != "ok" or "scan" not in m:
            continue
        curves.append(("Arrokoth", [p["rho_R"] for p in m["scan"]],
                       [p["dispersion"] for p in m["scan"]],
                       r["bulk_density_kg_m3"],
                       r["uniform_reference_dispersion"],
                       m["minimum"]["improvement_over_uniform_pct"]))
        break
    order = ["Itokawa", "67P", "Arrokoth", "Eros", "Bennu"]
    curves.sort(key=lambda c: order.index(c[0]))
    figure2(curves, "fig2_dispersion_curves.pdf")

    # ---- fig 3: mesh convergence ---------------------------------------
    pa, pb = [], []
    for r in runs:
        if "Itokawa" not in r["shape_model"]:
            continue
        if "--xsplit" in r["command"] or "--scale" in r["command"]:
            continue
        if abs(r["M_total_kg"] - 3.58e10) > 1e7:
            continue
        m = r["models"].get("head")
        if not m or m["status"] != "ok":
            continue
        (pa if abs(m["x_lo_km"] - 0.1702) < 1e-3 else pb).append(
            (r["facets_used"], m["minimum"]["rho_R"]))
    pa = sorted(set(pa)); pb = sorted(set(pb))
    ar = sorted({(r["facets_used"], r["models"]["head"]["minimum"]["rho_R"])
                 for r in arro if "arrokoth" in r["shape_model"]
                 and "--xsplit" in r["command"]})
    figure3([
        ("Itokawa, plane held fixed",
         [("$x_{\\rm split}=0.1702$ km", [p[0] for p in pa],
           [p[1] for p in pa], "o-"),
          ("$x_{\\rm split}=0.1608$ km", [p[0] for p in pb],
           [p[1] for p in pb], "s--")],
         "recovered $\\rho_{\\rm head}$ (kg m$^{-3}$)", 25.0),
        ("Arrokoth, plane held fixed at the neck",
         [("$x_{\\rm split}=-4.7863$ km", [p[0] for p in ar],
           [p[1] for p in ar], "^-")],
         "recovered $\\rho_{\\rm Wenu}$ (kg m$^{-3}$)", 5.0),
    ], "fig3_mesh_convergence.pdf")

    # ---- fig 7: Arrokoth mass sensitivity ------------------------------
    ms = root["batch_67P_arrokoth_bennu"]["arrokoth"][
        "mass_sensitivity_at_the_neck"]
    figure7([m["assumed_bulk_density"] for m in ms],
            [m["contrast"] for m in ms], "fig7_arrokoth_mass.pdf")


if __name__ == "__main__":
    main()
