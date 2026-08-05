#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure generation for SDGA Paper IV.

Layout follows the lesson learned in Paper III: figures are STACKED two-panel
portrait layouts sized for a single text column, so that fonts remain legible
(11-13 pt effective) when the figure is placed at \\textwidth. Side-by-side
landscape layouts scaled to textwidth shrink labels to ~5 pt and are unreadable
in print.

Figures produced
----------------
  fig3  Resolution behaviour: fractional deviation of the mean slope and of the
        normalized geopotential dispersion from their finest-mesh values.
        (Both panels share identical axes: that is the point of the figure.)
  fig4  Diagnostics: range sensitivity S and convergence residual C, per body.
  fig1  Ryugu per-facet validation against the Hayabusa2 products.
        REQUIRES the shape model and the archive file; skipped if absent.
  fig2  Bennu: latitudinal slope profile, and dependence on assumed density.
  fig5  Itokawa cross-version slope distributions, and the smoothing sequence.

Usage
-----
    python make_paper4_figures.py
    python make_paper4_figures.py --ryugu "Ryugu_SHAPE_SFM_49k_v20180804.obj" \
                                  --jaxa  "Gravity_SHAPE_SFM_49k_v20180804_7.63_1200a.txt"

Outputs fig1..fig5 as both .pdf (vector, for LaTeX) and .png (300 dpi).
All embedded numbers correspond to results/all_results.json,
key CORRECTED_after_centrifugal_signfix.
"""

import sys
import os
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# global style: sized so that text stays legible at \textwidth
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.size":        11.5,
    "axes.titlesize":   12.5,
    "axes.labelsize":   12.5,
    "xtick.labelsize":  10.5,
    "ytick.labelsize":  10.5,
    "legend.fontsize":  10,
    "figure.dpi":       120,
    "savefig.dpi":      300,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linewidth":   0.6,
    "axes.axisbelow":   True,
    "lines.linewidth":  1.8,
    "lines.markersize": 6,
})

FIGSIZE_STACKED = (6.4, 7.9)     # portrait; ~1:1 at a4 \textwidth (6.27 in)
                                 # so fonts render at their nominal size in print

# distinguishable in colour AND in greyscale (marker + linestyle differ)
STYLE = {
    "Ryugu":   dict(color="#1f77b4", marker="o", ls="-"),
    "Eros":    dict(color="#d62728", marker="s", ls="--"),
    "Itokawa": dict(color="#2ca02c", marker="^", ls="-."),
    "Ida":     dict(color="#9467bd", marker="D", ls=":"),
    "Gaspra":  dict(color="#ff7f0e", marker="v", ls=(0, (3, 1, 1, 1))),
}
ORDER = ["Ryugu", "Eros", "Itokawa", "Ida", "Gaspra"]

# --------------------------------------------------------------------------
# data (all_results.json -> CORRECTED_after_centrifugal_signfix)
# --------------------------------------------------------------------------
# facets, slope_avg [deg], pot_std_norm
SERIES = {
    "Ryugu":   [(3096, 12.174, 0.03153), (5780, 12.829, 0.03233),
                (11134, 13.688, 0.03295), (20014, 14.505, 0.03327),
                (42834, 15.700, 0.03347)],
    "Eros":    [(2097, 11.034, 0.03241), (6306, 11.691, 0.03345),
                (12155, 12.122, 0.03384), (21850, 12.705, 0.03403),
                (36030, 13.263, 0.03413)],
    "Itokawa": [(2183, 14.710, 0.08527), (6404, 15.167, 0.08577),
                (12272, 15.762, 0.08598), (21196, 16.304, 0.08608),
                (41369, 16.727, 0.08608)],
    "Gaspra":  [(2821, 12.413, 0.05929), (5240, 12.466, 0.05980),
                (12500, 12.564, 0.06009), (20773, 12.619, 0.06017),
                (32040, 12.641, 0.06017)],
    "Ida":     [(2141, 13.171, 0.03490), (5808, 13.560, 0.03572),
                (11804, 13.807, 0.03626), (20673, 13.918, 0.03641),
                (32040, 13.933, 0.03643)],
}

# Bennu latitudinal profile at rho = 1.194, Nf = 30561: (band centre, mean, median)
BENNU_LAT = [(5, 10.45, 9.70), (15, 12.02, 11.36), (25, 16.06, 15.48),
             (37.5, 19.10, 18.43), (52.5, 19.89, 19.32), (75, 17.83, 17.36)]
BENNU_LAT_LABELS = ["0-10", "10-20", "20-30", "30-45", "45-60", "60-90"]

# Bennu density sweep: rho, mean, eq(<20), mid(20-40), pol(>60)
BENNU_SWEEP = [(0.850, 25.55, 23.07, 29.38, 21.28),
               (1.000, 19.18, 14.57, 22.49, 19.43),
               (1.150, 16.06, 11.61, 18.19, 18.12),
               (1.194, 15.45, 11.19, 17.26, 17.81)]
# published comparison points (Scheeres et al. 2019): rho, mean slope
BENNU_PUB = [(0.85, 24.0), (1.15, 15.0)]

# Itokawa slope distribution, per cent of facets, matched resolution
# the ">90" bin is required: the spacecraft model puts 2.29% of facets there
# (overhangs / outward effective acceleration) and the radar model none, so
# omitting it would hide the very difference the figure is about.
ITO_BINS = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-90", ">90"]
ITO_GASKELL = [15.8, 25.2, 25.0, 15.7, 11.2, 4.8, 2.29]   # Nf = 4857
ITO_RADAR = [11.4, 23.6, 26.4, 21.0, 15.9, 1.7, 0.00]     # Nf = 3688

# smoothing: iterations, mean slope
SMOOTH = {
    "Itokawa": [(0, 15.81), (1, 14.82), (3, 14.00), (5, 13.56), (10, 12.90)],
    "Eros":    [(0, 13.65), (1, 12.23), (3, 11.35), (5, 10.92), (10, 10.30)],
}
ITO_RADAR_MEAN = 13.40    # independent radar model, unweighted mean slope

# Ryugu latitudinal comparison against the archive
RYUGU_LAT_BANDS = ["0-20", "20-40", "40-60", "60-90"]
RYUGU_OURS = [15.12, 12.23, 15.66, 21.25]
RYUGU_JAXA = [15.20, 12.40, 15.00, 20.80]


def sens_and_conv(vals):
    """Range sensitivity S and convergence residual C, both in per cent."""
    v = np.asarray(vals, float)
    S = (v.max() - v.min()) / v.mean() * 100.0
    C = abs(v[-1] - v[-2]) / v[-2] * 100.0
    return S, C


def save(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf and {name}.png")


# --------------------------------------------------------------------------
# fig3 : resolution behaviour, identical axes on both panels
# --------------------------------------------------------------------------
def fig_resolution():   # -> fig3 (third figure in the paper)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_STACKED)

    for body in ORDER:
        rows = SERIES[body]
        nf = np.array([r[0] for r in rows], float)
        sl = np.array([r[1] for r in rows], float)
        pv = np.array([r[2] for r in rows], float)
        st = STYLE[body]
        ax1.plot(nf, (sl - sl[-1]) / sl[-1] * 100.0, label=body, **st)
        ax2.plot(nf, (pv - pv[-1]) / pv[-1] * 100.0, label=body, **st)

    ylim = (-27, 3)
    for ax, title in ((ax1, r"(a) area-weighted mean slope $\bar{\theta}$"),
                      (ax2, r"(b) normalized geopotential dispersion $\tilde{\sigma}_U$")):
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.axhline(0.0, color="0.35", lw=1.0, zorder=1)
        ax.set_ylabel("deviation from finest mesh (%)")
        ax.set_title(title, loc="left")
    ax2.set_xlabel("number of facets $N_f$")
    ax2.text(0.985, 0.06, "identical vertical scale to (a)", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=9.5, color="0.3",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.6))
    ax1.legend(loc="lower right", ncol=2, framealpha=0.95)

    fig.suptitle("Behaviour of the two diagnostics under mesh refinement",
                 y=0.995, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save(fig, "fig3")


# --------------------------------------------------------------------------
# fig2 : S and C, grouped bars
# --------------------------------------------------------------------------
def fig_diagnostics():  # -> fig4
    S_sl, C_sl, S_pv, C_pv = [], [], [], []
    for body in ORDER:
        rows = SERIES[body]
        s1, c1 = sens_and_conv([r[1] for r in rows])
        s2, c2 = sens_and_conv([r[2] for r in rows])
        S_sl.append(s1); C_sl.append(c1); S_pv.append(s2); C_pv.append(c2)

    x = np.arange(len(ORDER))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_STACKED)

    for ax, a, b, title, ylab in (
            (ax1, S_sl, S_pv, "(a) range sensitivity $S$ over the full mesh range",
             "$S$ (%)"),
            (ax2, C_sl, C_pv, "(b) convergence residual $C$, final refinement step",
             "$C$ (%)")):
        ax.bar(x - w / 2, a, w, label=r"mean slope $\bar{\theta}$",
               color="#4c72b0", edgecolor="black", linewidth=0.6)
        ax.bar(x + w / 2, b, w, label=r"geopotential dispersion $\tilde{\sigma}_U$",
               color="#dd8452", edgecolor="black", linewidth=0.6, hatch="//")
        for xi, (va, vb) in enumerate(zip(a, b)):
            ax.text(xi - w / 2, va, f"{va:.1f}", ha="center", va="bottom", fontsize=9)
            ax.text(xi + w / 2, vb, f"{vb:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left")
        ax.set_ylim(0, max(a) * 1.22)
        ax.grid(axis="x", visible=False)
    ax1.legend(loc="upper right", framealpha=0.95)

    fig.suptitle("Resolution diagnostics for the two quantities",
                 y=0.995, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save(fig, "fig4")


# --------------------------------------------------------------------------
# fig3 : Ryugu per-facet validation (needs the two input files)
# --------------------------------------------------------------------------
def fig_ryugu(obj_path, jaxa_path):  # -> fig1
    if not (obj_path and jaxa_path and
            os.path.exists(obj_path) and os.path.exists(jaxa_path)):
        print("  fig1 (Ryugu) skipped: pass --ryugu <obj> --jaxa <txt> to generate it")
        return
    try:
        import polyhedral_gravity as pg
    except ImportError:
        print("  fig1 (Ryugu) skipped: polyhedral-gravity not installed")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from surface_slope_lib import load_obj, outward_normals, centroids

    rho_gcc, period_h = 1.2, 7.63

    J = []
    with open(jaxa_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 20:
                continue
            J.append((float(p[15]),))
    Jslope = np.array([j[0] for j in J])

    V, F = load_obj(obj_path)
    if len(F) != len(Jslope):
        print(f"  fig1 (Ryugu) skipped: facet count {len(F)} != archive {len(Jslope)}")
        return

    nf = outward_normals(V, F)
    C = centroids(V, F)
    Vm, Cm = V * 1000.0, C * 1000.0
    ext = np.linalg.norm(Vm.max(0) - Vm.min(0))
    pts = Cm - 1e-4 * ext * nf
    poly = pg.Polyhedron(polyhedral_source=(Vm.tolist(), F.tolist()),
                         density=rho_gcc * 1000.0,
                         normal_orientation=pg.NormalOrientation.OUTWARDS,
                         integrity_check=pg.PolyhedronIntegrity.DISABLE)
    res = pg.evaluate(poly, pts.tolist(), parallel=True)
    g = np.array([r[1] for r in res])
    om = 2 * math.pi / (period_h * 3600.0)
    rp = Cm.copy(); rp[:, 2] = 0.0
    g_eff = g + om ** 2 * rp
    gmag = np.linalg.norm(g_eff, axis=1)
    cos = np.einsum("ij,ij->i", -g_eff, nf) / np.clip(gmag, 1e-300, None)
    ours = np.degrees(np.arccos(np.clip(cos, -1, 1)))

    diff = ours - Jslope
    r = np.corrcoef(ours, Jslope)[0, 1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_STACKED)

    lim = (0, max(ours.max(), Jslope.max()) * 1.02)
    hb = ax1.hexbin(Jslope, ours, gridsize=70, bins="log",
                    cmap="viridis", mincnt=1, linewidths=0)
    ax1.plot(lim, lim, color="crimson", lw=1.3, ls="--", label="1:1")
    ax1.set_xlim(*lim); ax1.set_ylim(*lim)
    ax1.set_xlabel(r"archive slope (deg)")
    ax1.set_ylabel(r"this work, slope (deg)")
    ax1.set_title(f"(a) Ryugu, per facet ($N_f={len(F)}$), $r={r:.4f}$", loc="left")
    ax1.legend(loc="upper left", framealpha=0.95)
    cb = fig.colorbar(hb, ax=ax1, pad=0.02)
    cb.set_label("facets per bin")

    # data-driven range so that no residual is hidden (all outliers must be visible)
    lim2 = 1.05 * np.abs(diff).max()
    ax2.hist(diff, bins=160, range=(-lim2, lim2), color="#4c72b0", edgecolor="none")
    ax2.axvline(0.0, color="0.3", lw=1.0)
    ax2.set_yscale("log")
    ax2.set_xlabel("this work $-$ archive (deg)")
    ax2.set_ylabel("facets (log scale)")
    n_out5 = int((np.abs(diff) > 5).sum())
    ax2.set_title(f"(b) difference: RMS ${np.sqrt((diff**2).mean()):.2f}^\\circ$, "
                  f"{n_out5} of {len(diff)} facets beyond $5^\\circ$", loc="left")
    ax2.set_xlim(-lim2, lim2)

    fig.suptitle("Validation against the Hayabusa2 gravitational products",
                 y=0.995, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save(fig, "fig1")


# --------------------------------------------------------------------------
# fig2 : Bennu latitudinal profile and density dependence
# --------------------------------------------------------------------------
def fig_bennu():        # -> fig2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_STACKED)

    x = np.arange(len(BENNU_LAT))
    mean = [b[1] for b in BENNU_LAT]
    med = [b[2] for b in BENNU_LAT]
    ax1.plot(x, mean, color="#1f77b4", marker="o", ls="-", label="mean")
    ax1.plot(x, med, color="#8c8c8c", marker="s", ls="--", label="median")
    ax1.axhspan(11.0, 13.0, color="#2ca02c", alpha=0.16,
                label=r"published equatorial $\approx 12^\circ$")
    ax1.axhspan(17.0, 19.0, color="#d62728", alpha=0.13,
                label=r"published high-latitude $\approx 18^\circ$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(BENNU_LAT_LABELS)
    ax1.set_xlabel(r"$|\mathrm{latitude}|$ band (deg)")
    ax1.set_ylabel("slope (deg)")
    ax1.set_title(r"(a) Bennu latitudinal profile, $\varrho=1.194$, $N_f=30\,561$",
                  loc="left")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.95)

    rho = [s[0] for s in BENNU_SWEEP]
    ax2.plot(rho, [s[1] for s in BENNU_SWEEP], color="#1f77b4", marker="o",
             ls="-", label="global mean")
    ax2.plot(rho, [s[2] for s in BENNU_SWEEP], color="#2ca02c", marker="^",
             ls="--", label=r"equatorial, $|\mathrm{lat}|<20^\circ$")
    ax2.plot(rho, [s[3] for s in BENNU_SWEEP], color="#ff7f0e", marker="v",
             ls="-.", label=r"mid, $20$--$40^\circ$")
    ax2.plot(rho, [s[4] for s in BENNU_SWEEP], color="#9467bd", marker="D",
             ls=":", label=r"polar, $>60^\circ$")
    ax2.plot([p[0] for p in BENNU_PUB], [p[1] for p in BENNU_PUB],
             color="crimson", marker="*", ms=15, ls="none",
             label="published global mean")
    ax2.set_xlabel(r"assumed bulk density $\varrho$ (g cm$^{-3}$)")
    ax2.set_ylabel("slope (deg)")
    ax2.set_title("(b) dependence on the assumed bulk density", loc="left")
    ax2.legend(loc="upper right", fontsize=9, ncol=1, framealpha=0.95)

    fig.suptitle("Bennu: validation of the surface gravity field",
                 y=0.995, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save(fig, "fig2")


# --------------------------------------------------------------------------
# fig5 : cross-version distributions and smoothing
# --------------------------------------------------------------------------
def fig_cross():        # -> fig5
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_STACKED)

    x = np.arange(len(ITO_BINS))
    w = 0.38
    ax1.bar(x - w / 2, ITO_GASKELL, w, label=r"spacecraft model ($N_f=4\,857$)",
            color="#4c72b0", edgecolor="black", linewidth=0.6)
    ax1.bar(x + w / 2, ITO_RADAR, w, label=r"radar model ($N_f=3\,688$)",
            color="#dd8452", edgecolor="black", linewidth=0.6, hatch="//")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ITO_BINS)
    ax1.set_xlabel("slope bin (deg)")
    ax1.set_ylabel("facets (%)")
    ax1.set_title("(a) Itokawa slope distribution, two independent shape models",
                  loc="left")
    ax1.legend(loc="upper right", framealpha=0.95)
    ax1.grid(axis="x", visible=False)
    # annotation sits in the empty lower-right region, clear of every bar and of
    # the legend; the arrow points at the spacecraft ">90" bar it refers to
    ax1.annotate(r"radar model:" "\n"
                 r"no facet above $41^\circ$" "\n"
                 r"spacecraft: up to $149^\circ$",
                 xy=(6 - w / 2, ITO_GASKELL[-1] + 0.5), xytext=(4.52, 10.5),
                 fontsize=9.5, ha="left", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.8", lw=0.6),
                 arrowprops=dict(arrowstyle="->", lw=1.1, color="0.25",
                                 connectionstyle="arc3,rad=0.15"))

    for body, st in (("Itokawa", STYLE["Itokawa"]), ("Eros", STYLE["Eros"])):
        it = [s[0] for s in SMOOTH[body]]
        sl = [s[1] for s in SMOOTH[body]]
        ax2.plot(it, sl, label=body, **st)
    ax2.axhline(ITO_RADAR_MEAN, color="0.35", lw=1.2, ls=(0, (4, 2)))
    ax2.text(10.05, ITO_RADAR_MEAN + 0.12,
             "independent radar model\n" r"(Itokawa, $13.40^\circ$)",
             fontsize=9, ha="right", va="bottom", color="0.25")
    ax2.set_xlabel("Laplacian smoothing iterations")
    ax2.set_ylabel("mean slope (deg)")
    ax2.set_title("(b) smoothing at fixed facet count", loc="left")
    ax2.legend(loc="upper right", framealpha=0.95)

    fig.suptitle("Shape-model detail and the slope distribution",
                 y=0.995, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save(fig, "fig5")


def main():
    obj = jaxa = None
    if "--ryugu" in sys.argv:
        obj = sys.argv[sys.argv.index("--ryugu") + 1]
    if "--jaxa" in sys.argv:
        jaxa = sys.argv[sys.argv.index("--jaxa") + 1]

    print("generating Paper IV figures ...")
    fig_ryugu(obj, jaxa)     # fig1
    fig_bennu()              # fig2
    fig_resolution()         # fig3
    fig_diagnostics()        # fig4
    fig_cross()              # fig5
    print("done.")


if __name__ == "__main__":
    main()
