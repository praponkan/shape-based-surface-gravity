#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper3_figures.py - Paper III, all four figures
=====================================================
Fig 1  accuracy of the four methods on the real bodies      (Result 1)
Fig 2  which method wins where, plus (P2,P3) coverage        (Result 1)
Fig 3  the convexification bias, and why it cannot be fixed  (Result 2)
Fig 4  the prolate dominance is not rotational               (Result 3, exploratory)

INPUTS
    four_method_v2.json      four-method errors, real bodies, post-gate
    four_method_results.json  synthetic bodies (for Fig 2)
    convex_bias_results.json  true-vs-hull invariants and <g>
    bilobed_results.json      P2, P3, T and the bilobed metrics
    spin_data.csv             rotation periods, taxonomy, tumbler flags

GATES applied upstream
    P2 >= 0.10           below this w* is a 0/0 form and undefined; 42 -> 22
    Gauss bound eA <= 0  Paper I Sec. 8.2. Checked on every body, true mesh and
                         hull alike: all 42 pass, so nothing is removed by it.
                         It is an integrity check the pipeline passes, not a
                         filter. Its value here is diagnostic: an earlier,
                         uncalibrated solver put Sylvia at eA=+6.4% and Vesta at
                         eA=+82%, which the bound flagged as impossible; both
                         became physical (-2.0%, -4.1%) once the solver was
                         calibrated on a sphere.

RUN
    python make_paper3_figures.py
"""

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Larger, publication-legible fonts across all figures
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
})
from scipy import stats

C_RV, C_RA, C_EL, C_WB = "#c1272d", "#f7941e", "#5b9bd5", "#2e9e4f"
P2_GATE = 0.10
W_BLEND_ERR = 1.72          # Result 1 headline, used as the reference line

METHODS = [("R_V", "$R_V$\nalone", C_RV),
           ("R_A", "$R_A$\nalone", C_RA),
           ("ellipsoid", "fitted\nellipsoid", C_EL),
           ("w_blend", "$w$-blend\n(our law)", C_WB)]

RADAR_HINTS = ("radar", "1996hw1", "betulia", "rashalom", "kleo.v", "sf36",
               "mithra", "psyche", "kw4", "ky26", "ml14", "wt24", "yorp",
               "sk.obj", "ce26", "2008ev5", "1950da", "1994cc")


def short(name):
    base = name.split()[0]
    for junk in (".obj", ".mod", ".v", "_Radar"):
        base = base.replace(junk, "")
    star = "*" if any(h in name.lower() for h in RADAR_HINTS) else ""
    return base[:12] + star


def mean_abs(rows, key):
    v = np.array([r[key] for r in rows], float)
    v = v[np.isfinite(v)]
    return float(np.mean(np.abs(v))) if v.size else np.nan


def is_ellipsoidal(name):
    n = name.lower()
    return any(k in n for k in ["oblate", "prolate", "gentriax", "triax",
                                "ellips", "sphere"])


# ---------------------------------------------------------------------------
def fig1(real):
    rows = sorted(real, key=lambda r: r["P2"])
    # Stacked layout: sized so that at \textwidth the fonts stay ~10-13 pt.
    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(7.2, 9.2),
        gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.42})

    # ---- top: aggregate bars -------------------------------------------
    vals = [mean_abs(rows, k) for k, _, _ in METHODS]
    bars = axT.bar([lab for _, lab, _ in METHODS], vals,
                   color=[c for _, _, c in METHODS], edgecolor="k", lw=0.7)
    for b, v in zip(bars, vals):
        axT.text(b.get_x() + b.get_width()/2, v + max(vals)*0.02, f"{v:.2f}%",
                 ha="center", fontweight="bold", fontsize=14)
    axT.set_ylabel("mean |error| (%)", fontsize=14)
    axT.set_title(f"Accuracy against the polyhedron reference "
                  f"({len(rows)} models, $P_2 \\geq 0.10$)",
                  fontweight="bold", fontsize=14, pad=10)
    axT.set_ylim(0, max(vals) * 1.25)
    axT.tick_params(axis="both", labelsize=13)
    axT.grid(alpha=0.3, axis="y")
    axT.annotate("blend beats the\nfitted ellipsoid",
                 xy=(3, vals[3] + max(vals)*0.02),
                 xytext=(2.05, max(vals) * 0.62),
                 color=C_WB, fontweight="bold", ha="center", fontsize=13,
                 arrowprops=dict(arrowstyle="->", color=C_WB, lw=1.8))

    # ---- bottom: per-body bars -----------------------------------------
    x = np.arange(len(rows)); w = 0.2
    for i, (key, lab, col) in enumerate(METHODS):
        axB.bar(x + (i - 1.5)*w, [abs(r[key]) for r in rows], w,
                color=col, edgecolor="k", lw=0.4, label=lab.replace("\n", " "))
    axB.set_xticks(x)
    axB.set_xticklabels([short(r["name"]) for r in rows],
                        rotation=55, ha="right", fontsize=12)
    axB.set_ylabel("|error| (%)", fontsize=14)
    axB.set_title("Per-model error, ordered by $P_2$   (* = radar model)",
                  fontweight="bold", fontsize=14, pad=10)
    axB.tick_params(axis="y", labelsize=13)
    axB.legend(ncol=2, fontsize=12, framealpha=0.95)
    axB.grid(alpha=0.3, axis="y")

    fig.savefig("paper3_fig1_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote paper3_fig1_accuracy.png")
    return vals


# ---------------------------------------------------------------------------
def fig2(real, synth):
    ell = [r for r in synth if is_ellipsoidal(r["name"])]
    non = [r for r in synth if not is_ellipsoidal(r["name"])]
    # Stacked layout, sized so fonts stay ~11-13 pt at \textwidth.
    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(6.6, 8.8),
        gridspec_kw={"height_ratios": [1.0, 1.1], "hspace": 0.42})

    # ---- top: error by shape class -------------------------------------
    w = 0.38
    ve = [mean_abs(ell, k) for k, _, _ in METHODS]
    vn = [mean_abs(non, k) for k, _, _ in METHODS]
    for i, (key, lab, col) in enumerate(METHODS):
        axT.bar(i - w/2, ve[i], w, color=col, hatch="//", edgecolor="k", lw=0.7)
        axT.bar(i + w/2, vn[i], w, color=col, edgecolor="k", lw=0.7)
        axT.text(i - w/2, ve[i] + max(vn)*0.015, f"{ve[i]:.2f}",
                 ha="center", fontsize=12)
        axT.text(i + w/2, vn[i] + max(vn)*0.015, f"{vn[i]:.2f}", ha="center",
                 fontsize=13, fontweight="bold")
    axT.set_xticks(range(len(METHODS)))
    axT.set_xticklabels([lab for _, lab, _ in METHODS], fontsize=13)
    axT.set_ylabel("mean |error| (%)", fontsize=14)
    axT.set_title(f"Error by shape class\n"
                  f"(hatched: ellipsoidal, n={len(ell)};  "
                  f"solid: non-ellipsoidal, n={len(non)})",
                  fontweight="bold", fontsize=13, pad=8)
    axT.set_ylim(0, max(vn) * 1.22)
    axT.tick_params(axis="y", labelsize=13)
    axT.grid(alpha=0.3, axis="y")
    axT.annotate("ellipsoid is exact\nfor exact ellipsoids",
                 xy=(2 - w/2, ve[2] + max(vn)*0.02),
                 xytext=(0.75, max(vn)*0.80),
                 fontsize=12, color=C_EL, ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_EL, lw=1.6))
    axT.annotate("blend wins on\nreal-like shapes",
                 xy=(3 + w/2, vn[3] + max(vn)*0.02),
                 xytext=(2.55, max(vn)*0.52),
                 fontsize=12, color=C_WB, ha="center", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_WB, lw=1.6))

    # ---- bottom: shape-plane coverage ----------------------------------
    h_ell = axB.scatter([r["P2"] for r in ell], [r["P3"] for r in ell],
                        marker="s", s=34, c="#7fb2e5",
                        label="synthetic ellipsoidal")
    h_non = axB.scatter([r["P2"] for r in non], [r["P3"] for r in non],
                        marker="o", s=38, c=C_WB,
                        label="synthetic non-ellipsoidal")
    h_real = axB.scatter([r["P2"] for r in real], [r["P3"] for r in real],
                         marker="*", s=290, c=C_RV, edgecolor="k", lw=0.6,
                         label=f"real models (n={len(real)})", zorder=5)
    xx = np.linspace(0, max(r["P2"] for r in real + synth) * 1.05, 300)
    bd = xx**1.5 / np.sqrt(6)
    axB.plot(xx, bd, "k--", lw=1.0, alpha=0.6)
    h_bd, = axB.plot(xx, -bd, "k--", lw=1.0, alpha=0.6,
                     label=r"bound $|P_3| = P_2^{3/2}/\sqrt{6}$")
    axB.axhline(0, color="k", lw=0.7)
    h_gate = axB.axvline(P2_GATE, color=C_RV, ls="--", lw=1.2,
                         label=r"gate $P_2 = 0.10$")
    axB.set_xlabel("$P_2$   (deformation strength)", fontsize=14)
    axB.set_ylabel("$P_3$    (oblate < 0   /   prolate > 0)", fontsize=14)
    axB.set_title("Coverage of the shape plane", fontweight="bold",
                  fontsize=13, pad=42)
    axB.tick_params(axis="both", labelsize=13)
    axB.grid(alpha=0.3)

    # Legend split in two so neither block sits on the data:
    #   upper block above the axes, lower block in the empty lower-left.
    leg1 = axB.legend(handles=[h_ell, h_non], loc="lower left",
                      bbox_to_anchor=(0.0, 1.06), ncol=2,
                      fontsize=12, framealpha=0.95, borderaxespad=0.0)
    axB.add_artist(leg1)
    axB.legend(handles=[h_real, h_bd, h_gate], loc="lower left",
               fontsize=12, framealpha=0.95)

    # Oblate-body labels: the five oblate models sit almost on top of one
    # another near P2 ~ 0.1, so the names are placed in the empty upper-left
    # area, stacked and joined to their points by leader lines.
    obl = sorted([r for r in real if r["P3"] < 0], key=lambda r: r["P2"])
    label_x = [0.30, 0.30, 0.30, 0.30, 0.62]
    label_y = [0.78, 0.66, 0.54, 0.42, 0.30]
    for r, lx, ly in zip(obl, label_x, label_y):
        axB.annotate(short(r["name"]), xy=(r["P2"], r["P3"]),
                     xytext=(lx, ly), fontsize=11, color=C_RV,
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=C_RV,
                                     lw=0.8, alpha=0.75,
                                     shrinkA=2, shrinkB=4))

    fig.savefig("paper3_fig2_coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote paper3_fig2_coverage.png")


# ---------------------------------------------------------------------------
def fig3(cb):
    # fixed-GM convexification bias (the operational use case): the hull is
    # given the body's TRUE mass, so the bias reflects shape alone. Values are
    # produced by paper3_audit.py and stored in fixedgm_bias.json.
    # Match names case-insensitively (audit lowercases them).
    fixedgm_raw = json.load(open("fixedgm_bias.json"))
    fixedgm = {k.lower(): v for k, v in fixedgm_raw.items()}
    clear = [r for r in cb if r["true"]["P2"] >= P2_GATE
             and r["name"].lower() in fixedgm]
    cd = np.array([r["bias"]["convex_deficiency"] for r in clear])
    gb = np.array([fixedgm[r["name"].lower()] for r in clear])
    keep = np.isfinite(cd) & np.isfinite(gb)
    names = [short(r["name"]) for r, k in zip(clear, keep) if k]
    cd, gb = cd[keep], gb[keep]

    r_p, p_p = stats.pearsonr(cd, gb)
    A = np.vstack([cd, np.ones(len(cd))]).T
    coef, *_ = np.linalg.lstsq(A, gb, rcond=None)
    r2 = 1 - np.sum((gb - A@coef)**2) / np.sum((gb - gb.mean())**2)

    # Stacked layout, sized so fonts stay ~12 pt at \textwidth.
    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(6.8, 9.0),
        gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.34})

    # ---- top: bias vs convex deficiency --------------------------------
    axT.axhline(-W_BLEND_ERR, color="k", lw=1.3, alpha=0.6,
                label=f"blend's own error ({W_BLEND_ERR:.2f}%)")
    axT.scatter(cd, gb, s=110, c=C_RV, edgecolor="k", lw=0.7, zorder=3)
    xs = np.linspace(0, cd.max()*1.05, 100)
    axT.plot(xs, coef[0]*xs + coef[1], "k--", lw=1.6,
             label=f"fit: {coef[0]:.1f}·CD {coef[1]:+.1f}   $R^2$={r2:.3f}")
    # Labels for the most deformed models, fanned out to avoid collisions.
    lab_pts = sorted([(n, x, y) for n, x, y in zip(names, cd, gb)
                      if y < -6 or x > 0.18], key=lambda t: t[1])
    lab_xy = [(0.055, -8.2), (0.055, -9.9), (0.055, -11.6),
              (0.055, -13.3), (0.20, -14.9)]
    for (n, x, y), (lx, ly) in zip(lab_pts, lab_xy):
        axT.annotate(n, xy=(x, y), xytext=(lx, ly), fontsize=11,
                     ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color="0.35",
                                     lw=0.8, alpha=0.8, shrinkA=2, shrinkB=5))
    axT.set_xlabel(r"convex deficiency,   $1 - V_{\rm true}/V_{\rm hull}$",
                   fontsize=14)
    axT.set_ylabel(r"error in $\langle g \rangle$ (%)", fontsize=14)
    axT.set_title(f"Convex models underestimate the gravity, and the bias\n"
                  f"is almost perfectly predictable  (r = {r_p:+.3f})",
                  fontweight="bold", fontsize=13, pad=10)
    axT.tick_params(axis="both", labelsize=13)
    axT.legend(fontsize=12, loc="upper right", framealpha=0.95)
    axT.grid(alpha=0.3)

    # ---- bottom: magnitude comparison ----------------------------------
    labels = ["blend's own\nerror", "convex bias\n(mean)", "convex bias\n(worst)"]
    vals = [W_BLEND_ERR, float(np.abs(gb).mean()), float(np.abs(gb).max())]
    bars = axB.bar(labels, vals, color=[C_WB, C_RV, "#7a1518"],
                   edgecolor="k", lw=0.7, width=0.6)
    for b, v in zip(bars, vals):
        axB.text(b.get_x() + b.get_width()/2, v + max(vals)*0.02, f"{v:.2f}%",
                 ha="center", fontweight="bold", fontsize=14)
    axB.set_ylabel(r"error in $\langle g \rangle$ (%)", fontsize=14)
    axB.set_title(f"The bias is {vals[1]/vals[0]:.1f}× the signal it would "
                  f"measure", fontweight="bold", fontsize=13, pad=10)
    axB.set_ylim(0, max(vals)*1.42)
    axB.tick_params(axis="both", labelsize=13)
    axB.grid(alpha=0.3, axis="y")
    axB.text(0.055, 0.62,
             "the fit is tight,\n"
             "but the correction\n"
             "is circular: it needs\n"
             r"$V_{\rm true}$, and a convex"
             "\nmodel has no way\n"
             "to provide it",
             transform=axB.transAxes, ha="left", va="center",
             fontsize=11.5, style="italic",
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff3cd", ec="#856404",
                       alpha=0.95))

    fig.savefig("paper3_fig4_convexbias.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote paper3_fig4_convexbias.png")
    return r_p, r2, vals[1], vals[2]


# ---------------------------------------------------------------------------
def fig4(bil, spin_csv):
    by = {r["name"]: r for r in bil}
    name_map = {
        'Kleopatra': 'Kleopatra Ostro', '1996HW1': '1996hw1',
        'Eros': 'Eros Gaskell 50k poly', 'Ida': 'Ida Thomas',
        'Itokawa': 'Itokawa Hayabusa 50k poly',
        'Geographos': 'Geographos Radar-based',
        'Toutatis': 'Toutatis Radar-based, hi-res',
        'Bacchus': 'Bacchus Radar-based',
        'Castalia': 'Castalia Radar-based',
        'Mithra': 'Mithra.v1.PA.prograde.mod',
        'Nereus': 'Nereus Radar-based', 'Gaspra': 'Gaspra Thomas',
        'Eugenia': 'Eugenia Hanus 2013', 'Kalliope': 'Kalliope Kaasalainen',
        'Steins': 'Steins Radar', 'Betulia': 'betulia',
        'Sylvia': 'Sylvia Kaasalainen', 'RaShalom': 'rashalom.obj',
        'Lutetia': 'Lutetia Farnham 50k poly'}

    def lookup(nm):
        """Match a spin-table name to a mesh name, tolerating the archive
        suffixes ('Radar-based, mid-res', '50k poly', '.obj')."""
        bn = name_map.get(nm)
        if bn and bn in by:
            return by[bn]
        if bn:                       # fall back to a prefix match
            for k, v in by.items():
                if k.startswith(bn):
                    return v
        for k, v in by.items():      # last resort: first word
            if k.lower().split()[0].split('.')[0] == nm.lower():
                return v
        return None

    rows = []
    with open(spin_csv) as f:
        for s in csv.DictReader(f):
            b = lookup(s["name"])
            if b is None:
                print(f"    [warn] no mesh for {s['name']}")
                continue
            rows.append(dict(name=s["name"], P=float(s["P_hr"]), taxo=s["taxo"],
                             tumbler=int(s["tumbler"]), T=b["T"], P2=b["P2"]))
    ok = [r for r in rows if not r["tumbler"]]
    tum = [r for r in rows if r["tumbler"]]

    w = 1.0 / np.array([r["P"] for r in ok])
    T = np.array([r["T"] for r in ok])
    r_p, p_p = stats.pearsonr(w, T)

    # Stacked layout, sized so fonts stay ~12 pt at \textwidth.
    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(6.9, 9.4),
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.36})

    tax_col = {"S": C_RA, "Sq": C_RA, "Sk": C_RA, "C": "#333333",
               "P": "#666666", "M": "#9467bd", "E": "#17becf", "Xe": "#17becf"}
    # Hand-tuned label offsets: several bodies cluster near P = 5 h.
    off = {"Castalia": (-42, 10), "Kalliope": (8, -14), "Ida": (-26, -16),
           "Sylvia": (-40, 12), "Geographos": (8, 12), "Eros": (8, -2),
           "Kleopatra": (8, 12), "Eugenia": (-52, -6), "Steins": (8, -14),
           "Betulia": (8, 6), "Gaspra": (8, -14), "Lutetia": (8, -14),
           "1996HW1": (8, 8), "Itokawa": (8, -14), "Bacchus": (8, 8),
           "Nereus": (8, -14), "RaShalom": (-30, -18)}
    for r in ok:
        axT.scatter(r["P"], r["T"], s=125, c=tax_col.get(r["taxo"], "gray"),
                    edgecolor="k", lw=0.7, zorder=3)
        axT.annotate(r["name"], (r["P"], r["T"]), textcoords="offset points",
                     xytext=off.get(r["name"], (8, -3)), fontsize=11)
    for r in tum:
        axT.scatter(r["P"], r["T"], s=125, facecolor="none", edgecolor="gray",
                    lw=1.4, marker="s", zorder=3)
        axT.annotate(r["name"] + " (tumbler)", (r["P"], r["T"]),
                     textcoords="offset points", xytext=(-30, -22),
                     fontsize=11, color="gray")
    for a, b in (("Kleopatra", "Eugenia"), ("Bacchus", "RaShalom")):
        pa = next((r for r in ok if r["name"] == a), None)
        pb = next((r for r in ok if r["name"] == b), None)
        if pa and pb:
            axT.plot([pa["P"], pb["P"]], [pa["T"], pb["T"]], color="red",
                     lw=1.3, ls=":", alpha=0.8, zorder=2)
    axT.set_xscale("log")
    axT.axhline(0, color="k", lw=0.7)
    axT.set_xlabel("rotation period (h)      faster $\\leftarrow$", fontsize=14)
    axT.set_ylabel("$T$   ($+1$ prolate,  $-1$ oblate)", fontsize=14)
    axT.set_title(f"Spin does not explain the shape\n"
                  f"r = {r_p:+.3f},  p = {p_p:.2f}   "
                  f"(n = {len(ok)}, tumblers excluded)",
                  fontweight="bold", fontsize=13, pad=10)
    axT.set_ylim(-0.95, 1.32)
    axT.tick_params(axis="both", labelsize=13)
    axT.grid(alpha=0.3)
    axT.text(0.02, 0.03, "red dotted: same period, opposite shape",
             transform=axT.transAxes, fontsize=11.5, color="red")

    S = [r["T"] for r in ok if r["taxo"] in ("S", "Sq", "Sk")]
    C = [r["T"] for r in ok if r["taxo"] in ("C", "P")]
    oth = [r["T"] for r in ok if r["taxo"] not in ("S", "Sq", "Sk", "C", "P")]
    parts = [(f"S-type\n(n={len(S)})", S, C_RA),
             (f"C/P-type\n(n={len(C)})", C, "#333333"),
             (f"other\n(n={len(oth)})", oth, "#999999")]
    rng = np.random.default_rng(1)
    for i, (lab, v, col) in enumerate(parts):
        if not v:
            continue
        axB.scatter(np.full(len(v), i) + rng.uniform(-0.07, 0.07, len(v)), v,
                    s=125, c=col, edgecolor="k", lw=0.7, zorder=3)
        axB.hlines(np.mean(v), i - 0.26, i + 0.26, color=col, lw=3.0)
    if S and C:
        u, p = stats.mannwhitneyu(S, C, alternative="greater")
        axB.set_title("Taxonomy tracks the shape — but the confound is total\n"
                      f"S vs C/P:  Mann-Whitney p = {p:.3f}   "
                      "[exploratory, not pre-registered]",
                      fontweight="bold", fontsize=13, pad=10)
    axB.set_xticks(range(len(parts)))
    axB.set_xticklabels([p[0] for p in parts], fontsize=13)
    axB.axhline(0, color="k", lw=0.7)
    axB.set_ylim(-1.15, 1.15)
    axB.set_xlim(-0.55, 2.55)
    axB.set_ylabel("$T$", fontsize=14)
    axB.tick_params(axis="y", labelsize=13)
    axB.grid(alpha=0.3, axis="y")
    # Caveat box placed below the axes, so it cannot overlap any data point.
    axB.text(0.5, -0.30,
             "S-types here are small NEAs measured by radar; C/P-types are large main-belt\n"
             "bodies from lightcurve+AO. Composition, size, population and observing method\n"
             "all move together — they cannot be separated at n = %d." % len(ok),
             transform=axB.transAxes, ha="center", va="top",
             fontsize=11.5, style="italic",
             bbox=dict(boxstyle="round,pad=0.45", fc="#fff3cd", ec="#856404",
                       alpha=0.95))

    fig.savefig("paper3_fig5_spin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote paper3_fig5_spin.png")
    return r_p, p_p, len(ok)


def main():
    # four_method_v2.json holds the full 22-model real sample with the
    # exact index-integral ellipsoid column. If it is not present, fall
    # back to four_method_results.json (which carries the ellipsoid column
    # for the subset used in Figures 1-2).
    try:
        real = json.load(open("four_method_v2.json"))["results"]["real"]
    except FileNotFoundError:
        real = json.load(open("four_method_results.json"))["results"]["real"]
    try:
        syn = json.load(open("four_method_results.json"))["results"]["synthetic"]
    except Exception:
        syn = []
    cb = json.load(open("convex_bias_results.json"))["results"]
    bil = json.load(open("bilobed_results.json"))["results"]

    n_ob = sum(1 for r in real if r["P3"] < 0)
    print(f"loaded: {len(real)} real ({len(real)-n_ob} prolate, {n_ob} oblate), "
          f"{len(syn)} synthetic, {len(cb)} convex-bias, {len(bil)} bilobed\n")

    v = fig1(real)
    if syn:
        fig2(real, syn)
    else:
        print("  [skip] fig2 needs the synthetic set")
    r_p, r2, mb, wb = fig3(cb)
    rs, ps, n = fig4(bil, "spin_data.csv")

    print("\n--- numbers for the manuscript ---")
    print(f"  Fig1  R_V {v[0]:.2f}%  R_A {v[1]:.2f}%  ellipsoid {v[2]:.2f}%  "
          f"w-blend {v[3]:.2f}%")
    print(f"  Fig3  bias mean {mb:.2f}%  worst {wb:.2f}%  r={r_p:+.3f}  R2={r2:.3f}"
          f"  ratio {mb/W_BLEND_ERR:.1f}x")
    print(f"  Fig4  corr(spin,T) r={rs:+.3f} p={ps:.3f}  n={n}")


if __name__ == "__main__":
    main()
