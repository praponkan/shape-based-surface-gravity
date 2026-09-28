import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

def neck(p):
    r = [x for x in json.load(open(p))["results"] if x["plane"] == "neck"][0]
    return r["clean"]["contrast"], r["all"]["contrast"]

rho = np.array([235.0, 250.0, 400.0, 500.0])
clean, raw = np.array([neck(p) for p in ("arro_235.json", "arro_250.json",
                                         "arrokoth_clean.json",
                                         "arro_500.json")]).T

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.axvspan(200, 290, color="0.91", zorder=0, lw=0)
ax.axhline(1.0, color="0.45", lw=0.9, ls=":", zorder=1)
ax.plot(rho, raw, ls="--", marker="s", mfc="white", mec="0.45", color="0.55",
        ms=7, mew=1.4, lw=1.5, zorder=2,
        label="all facets, including 455 inside the body")
ax.plot(rho, clean, ls="-", marker="o", color="#1f4e79", ms=7.5, lw=2.2,
        zorder=3, label="buried facets removed")
for x, y, dy, ha in ((235, clean[0], 12, "left"), (250, clean[1], -19, "left"),
                     (400, clean[2], -19, "center"), (500, clean[3], 12, "right")):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                xytext=(0, dy), ha=ha, fontsize=9.5, color="#1f4e79")
ax.set_xlim(205, 545)
ax.set_ylim(0.895, 1.10)
ax.set_xlabel("assumed bulk density (kg m$^{-3}$)")
ax.set_ylabel(r"recovered contrast  $\varrho_{\rm Wenu}/\varrho_{\rm rest}$")
ax.set_title("Arrokoth: the assumed mass sets the answer", fontsize=11.5)
ax.text(247, 0.902, "excluded by\nmutual gravity", ha="center", va="bottom",
        fontsize=9, color="0.35")
ax.text(541, 1.003, "equal densities", ha="right", va="bottom", fontsize=9,
        color="0.45")
ax.legend(loc="upper right", frameon=False, fontsize=9.5, handlelength=2.6,
          borderaxespad=0.8, labelspacing=0.7)
fig.tight_layout()
fig.savefig("fig05_arrokoth_mass.pdf")
print("clean:", clean.round(4), " crosses unity:",
      bool(clean.max() > 1 > clean.min()))
