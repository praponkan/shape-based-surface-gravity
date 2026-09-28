"""Figure 3: dispersion against region density, five bodies, each normalized.

    cd data && python ../code/make_fig03.py

The Arrokoth curve is read from arrokoth_scan_clean.json, the run with the
455 buried facets excluded, so that the figure and the tables describe the
same run.
"""
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

d = [r for r in json.load(open("paperV_final.json"))
     if "meshcache" in r.get("command", "")
     and "--decimator cluster" not in r["command"]]


def pick(frag, x, mass=None, tol=5e-4):
    for r in d:
        if frag not in r["shape_model"] or "--fracsplit" in r["command"]:
            continue
        if mass and abs(r["M_total_kg"] - mass) > 1e-3 * mass:
            continue
        m = r["models"].get("head")
        if m and m["status"] == "ok" and abs(m["x_lo_km"] - x) < tol \
                and m.get("scan"):
            return m, r
    raise SystemExit(f"no run for {frag}")


curves = []
for lab, frag, x, mass, col, ls, mk in (
        ("Itokawa", "Itokawa", 0.1608, 3.58e10, "#1f4e79", "-", "o"),
        ("Eros", "Eros", 4.4700, None, "#4d4d4d", "--", "s"),
        ("Bennu", "Bennu", 0.0, None, "#8c8c8c", ":", "^"),
        ("67P", "cg_mspcd", 0.9517, None, "#9c6b30", "-.", "D")):
    m, r = pick(frag, x, mass)
    rho = np.array([p["rho_R"] for p in m["scan"]])
    dis = np.array([p["dispersion"] for p in m["scan"]])
    curves.append((lab, rho / r["bulk_density_kg_m3"],
                   dis / r["uniform_reference_dispersion"], col, ls, mk,
                   m["minimum"]["improvement_over_uniform_pct"]))

a = json.load(open("arrokoth_scan_clean.json"))["curves"]["400"]
curves.append(("Arrokoth", np.array(a["rho_grid"]) / 400.0,
               np.array(a["dispersion"]) / a["dispersion_uniform"],
               "#c0392b", (0, (5, 1, 1, 1, 1, 1)), "v", a["improvement_pct"]))

fig, ax = plt.subplots(figsize=(7.0, 4.8))
ax.axhline(1.0, color="0.6", lw=0.8, ls=":", zorder=1)
for lab, x, y, col, ls, mk, imp in curves:
    k = (x >= 0.45) & (x <= 2.15)
    lw = 2.2 if lab in ("Itokawa", "Arrokoth") else 1.5
    i = int(np.nanargmin(y))
    ax.plot(x[k], y[k], color=col, ls=ls, lw=lw, zorder=3)
    ax.plot([x[i]], [y[i]], marker=mk, ms=8, mfc="white", mec=col, mew=1.8,
            ls="none", zorder=5,
            label=f"{lab}  ({imp:.2f} per cent)".replace(" per cent", "%"))

ax.set_xlim(0.45, 2.15)
ax.set_ylim(0.80, 1.22)
ax.set_xlabel(r"region density / bulk density")
ax.set_ylabel(r"dispersion / uniform-density dispersion")
ax.set_title("One deep well and four shallow ones", fontsize=11.5)
leg = ax.legend(loc="lower left", frameon=True, framealpha=0.0,
                edgecolor="0.55", fontsize=9.5, labelspacing=0.6,
                handlelength=2.8, borderpad=0.7,
                title="body (marker at the minimum; improvement)",
                title_fontsize=9)
leg.get_frame().set_facecolor("none")
for h, (lab, x, y, col, ls, mk, imp) in zip(leg.legend_handles, curves):
    h.set_linestyle(ls)
    h.set_color(col)
    h.set_linewidth(2.0)
fig.tight_layout()
fig.savefig("fig03_dispersion_curves.pdf")
for lab, x, y, c, ls, mk, imp in curves:
    print(f"  {lab:<9} min at rho/bulk {x[int(np.nanargmin(y))]:.3f}  "
          f"depth {1 - y.min():.4f}  improvement {imp:.3f}%")
