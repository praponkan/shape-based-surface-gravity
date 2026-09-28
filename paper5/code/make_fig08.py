#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 8: dispersion improvement against dividing plane, Itokawa.

    cd data && python ../code/make_fig08.py

Reads paperV_final.json. Shows that the improvement rises monotonically to
the end of the sweep, and that the maximum over the compositionally
admissible set is attained at the boundary of that set rather than at an
interior optimum: the two vertical bands are the block-density bound at
0.1935 km and the grain-density ceiling at 0.2069 km.
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
sw = {}
for r in d:
    if "Itokawa" not in r["shape_model"] or r["facets_used"] != 49152:
        continue
    if abs(r["M_total_kg"] - 3.58e10) > 1e7 or "--scale" in r["command"]:
        continue
    m = r["models"].get("head")
    if m and m["status"] == "ok":
        sw.setdefault(round(m["x_lo_km"], 4),
                      m["minimum"]["improvement_over_uniform_pct"])
x = np.array(sorted(sw))
I = np.array([sw[k] for k in x])

fig, ax = plt.subplots(figsize=(7.0, 4.4))
ax.axvspan(0.1935, 0.2069, color="0.93", zorder=0, lw=0)
ax.axvspan(0.2069, 0.25, color="0.86", zorder=0, lw=0)
ax.axvline(0.1935, color="0.45", lw=1.0, ls="--", zorder=1)
ax.axvline(0.2069, color="0.30", lw=1.0, ls="-", zorder=1)

ax.plot(x, I, "o-", color="#1f4e79", ms=5.5, lw=1.8, zorder=3)
k = np.argmin(abs(x - 0.1608))
ax.plot([x[k]], [I[k]], "o", ms=11, mfc="none", mec="#c0392b", mew=2,
        zorder=4)
ax.annotate("detected neck\n16.89%", (x[k], I[k]), textcoords="offset points",
            xytext=(6, -34), fontsize=9.5, color="#c0392b")
ax.plot([0.1935], [21.49], "s", ms=7, color="0.25", zorder=4)
ax.annotate("block-density bound\n21.5%", (0.1935, 21.49),
            textcoords="offset points", xytext=(-8, 10), ha="right",
            fontsize=9.5, color="0.25")

ax.text(0.2002, 13.6, "excluded by the\nblock-density bound", fontsize=8.5,
        color="0.35", ha="center")
ax.text(0.2285, 13.6, "excluded by the\ngrain-density ceiling", fontsize=8.5,
        color="0.25", ha="center")

ax.set_xlim(0.125, 0.245)
ax.set_ylim(11.5, 28)
ax.set_xlabel(r"dividing plane $x_{\rm split}$ (km)")
ax.set_ylabel("improvement in dispersion over uniform (%)")
ax.set_title("The objective prefers the boundary, not the neck", fontsize=11.5)
fig.tight_layout()
fig.savefig("fig08_improvement_vs_plane.pdf")
print(f"wrote fig08_improvement_vs_plane.pdf  ({len(x)} planes, "
      f"I from {I.min():.3f} to {I.max():.3f}%, monotonic: "
      f"{bool(np.all(np.diff(I) > 0))})")
