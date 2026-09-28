#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 6 in one command: collect the cross-section profiles, then draw them.

    python run_profiles.py

Five --profile runs, which evaluate no gravity and take seconds each, then the
figure is drawn from the recorded data. The panel for each body marks the
saddle the detector selected and, where the two differ, the flank it rejected.

Options
-------
    --record FILE    profile JSON (default profiles.json)
    --script FILE    solver to invoke (default density_estimate.py)
    --out FILE       figure file (default fig6_profiles.pdf)
    --figure-only    skip the runs and draw from an existing record
    --list           print the job list and exit
    --force          rerun even if the record already has the entry
"""

import json
import os
import subprocess
import sys

RECORD = "profiles.json"
SCRIPT = "density_estimate.py"
OUT = "fig6_profiles.pdf"

BODIES = [
    ("Itokawa", "Itokawa Hayabusa 50k poly.obj", "3.58e10", "12.1324"),
    ("Eros", "Eros Gaskell 50k poly.obj", "6.687e15", "5.27"),
    ("Bennu", "Bennu_v20_200k.obj", "7.330e10", "4.296061"),
    ("67P", "cg_mspcd_shap2_001m_cart.obj", "9.982e12", "12.4043"),
    ("Arrokoth", "arrokoth_porter_2024_v01.obj", "1.6495e15", "15.92"),
]

# only these three go in the figure: one clean saddle, one rejected flank, one
# refusal. 67P and Arrokoth are collected as well so the record is complete.
PANELS = ["Itokawa", "Eros", "Bennu"]


def jobs():
    out = []
    for name, obj, mtotal, period in BODIES:
        out.append({
            "label": name,
            "args": [obj, "--mtotal", mtotal, "--period", period,
                     "--target", "50000", "--profile"],
        })
    return out


def already(record, script, args, rec):
    if not os.path.exists(record):
        return False
    try:
        data = json.load(open(record, encoding="utf-8"))
    except Exception:
        return False
    want = " ".join([script] + args + ["--profilerecord", rec])
    return any(r.get("command") == want for r in data)


def collect(record, script, force):
    for i, j in enumerate(jobs(), start=1):
        cmd = [sys.executable, script] + j["args"] + \
            ["--profilerecord", record]
        if not force and already(record, script, j["args"], record):
            print(f"[{i}/{len(BODIES)}] {j['label']}: already recorded")
            continue
        print(f"\n[{i}/{len(BODIES)}] {j['label']}")
        rc = subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode
        print("  done" if rc == 0 else f"  FAILED with code {rc}")


def draw(record, out):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.5, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "legend.frameon": False, "figure.dpi": 200,
    })
    C_OK, C_BAD, C_GREY = "#1f4e79", "#b3541e", "#8c8c8c"

    data = json.load(open(record, encoding="utf-8"))
    by_body = {}
    for r in data:
        for name, obj, _, _ in BODIES:
            if os.path.basename(obj) == os.path.basename(r["shape_model"]):
                by_body[name] = r
    missing = [b for b in PANELS if b not in by_body]
    if missing:
        sys.exit(f"no profile recorded for: {', '.join(missing)}")

    fig, axes = plt.subplots(1, len(PANELS), figsize=(8.0, 3.2))
    for ax, name in zip(axes, PANELS):
        r = by_body[name]
        x = np.array(r["x_km"], float)
        w = np.array(r["half_width_km"], float)
        nk = r["neck"]
        ax.plot(x, w, "-", color="black", lw=1.1, zorder=3)
        ax.fill_between(x, 0, w, color=C_GREY, alpha=0.12, lw=0, zorder=1)

        if nk["found"]:
            ax.axvline(nk["x_neck_km"], color=C_OK, lw=1.1, ls="--", zorder=2)
            ax.plot([nk["x_neck_km"]], [nk["half_width_km"]], "o", ms=5,
                    color=C_OK, mfc="white", mew=1.4, zorder=5)
            xr = (nk["x_neck_km"] - x.min()) / (x.max() - x.min())
            ax.annotate(f"saddle,\nprominence {nk['prominence']:.3f}",
                        (nk["x_neck_km"], nk["half_width_km"]),
                        textcoords="offset points",
                        xytext=(-8 if xr > 0.55 else 8, -30),
                        ha="right" if xr > 0.55 else "left",
                        fontsize=7, color=C_OK, linespacing=1.35)
        if nk["flank_rejected"] or not nk["found"]:
            xa = nk["absolute_interior_minimum_x_km"]
            wa = nk["absolute_interior_minimum_half_width_km"]
            ax.plot([xa], [wa], "x", ms=7, color=C_BAD, mew=1.6, zorder=5)
            lab = ("smallest interior half-width:\nflank, rejected"
                   if nk["found"] else
                   "smallest interior half-width:\nnot a saddle")
            # park the label in the upper corner on the same side as the
            # marker and run a thin leader to it, so it never sits on the
            # profile
            xr = (xa - x.min()) / (x.max() - x.min())
            cx, ha = (0.97, "right") if xr > 0.5 else (0.03, "left")
            ax.annotate(lab, xy=(xa, wa), xycoords="data",
                        xytext=(cx, 0.955), textcoords="axes fraction",
                        ha=ha, va="top", fontsize=7, color=C_BAD,
                        linespacing=1.35,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=C_BAD,
                                        shrinkA=2, shrinkB=4))

        sub = (f"{nk['n_accepted']} of {nk['n_local_minima']} local minima "
               f"accepted" if nk["found"] else
               f"no saddle accepted\n({nk['n_local_minima']} local minima, "
               f"prominence $<$ {nk['min_prominence_threshold']:.2f})")
        ax.set_title(f"{name}\n{sub}", fontsize=8.2, pad=6,
                     linespacing=1.45)
        ax.set_xlabel("$x$ (km)")
        ax.set_ylim(0, 1.45 * float(np.nanmax(w)))
        pad = 0.06 * (x.max() - x.min())
        ax.set_xlim(x.min() - pad, x.max() + pad)
    axes[0].set_ylabel("cross-section half-width (km)")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


def main():
    argv = sys.argv[1:]

    def opt(n, d=None):
        return argv[argv.index(n) + 1] if n in argv else d

    record = opt("--record", RECORD)
    script = opt("--script", SCRIPT)
    out = opt("--out", OUT)

    if "--list" in argv:
        for j in jobs():
            print(f"  {j['label']:<10} {' '.join(j['args'])}")
        return
    if "--figure-only" not in argv:
        if not os.path.exists(script):
            sys.exit(f"solver not found: {script}")
        collect(record, script, "--force" in argv)
    draw(record, out)


if __name__ == "__main__":
    main()
