#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Archive the full dispersion curve for Arrokoth with the buried facets removed.

    python arrokoth_scan_clean.py arrokoth_porter_2024_v01.obj

Writes arrokoth_scan_clean.json.

Why this exists. Figure 3 plots the dispersion against region density for all
five bodies. Its Arrokoth curve comes from the run that keeps the 455 facets
lying inside the opposite shell, because rerun_arrokoth_clean.py archived only
the values at the minimum and not the curve itself. Everything else about
Arrokoth in the paper excludes those facets, so the figure and the tables
disagree by the difference between an improvement of 0.88 and one of 1.27 per
cent. This records the curve so that the figure can be drawn from the same
run as the tables.

The buried facets are identified exactly as in rerun_arrokoth_clean.py, whose
functions this imports rather than reimplements. The scan is evaluated at all
four assumed bulk densities in one pass: the unit potentials are computed once
and the four densities differ only in the sums taken over them, so the three
extra densities are nearly free.
"""

import json
import sys
import time

import numpy as np

import density_estimate as de
from rerun_arrokoth_clean import find_buried
from surface_slope_lib import load_obj, outward_normals

PLANE = -4.7863          # km, the detected neck
PERIOD = 15.92           # h, Spencer et al. (2020)
SCAN = (25.0, 3000.0, 5.0)
DENSITIES = [235.0, 250.0, 400.0, 500.0]


def curve(V, F, keep, M_tot, x_split, scan):
    """Dispersion against region density, over the retained facets only.

    Every line here follows rerun_arrokoth_clean.invert exactly: the same
    absolute volumes, the same triangle areas computed from the cross
    product, and the potentials evaluated at every facet and masked
    afterwards rather than at the retained facets alone. The two differ only
    in that this one keeps the whole curve. If they disagreed, the curve
    would not describe the run the tables report, which is the one thing
    this script exists to prevent.
    """
    V_tot = abs(de.closed_volume(V, F))
    Vr, Fr = de.clip_halfspace(V, F, 0, x_split, True)
    V_R = abs(de.closed_volume(Vr, Fr))

    nf = outward_normals(V, F)
    C = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    ext = float(np.linalg.norm(V.max(0) - V.min(0)))
    pts_m = (C - 1e-4 * ext * nf) * 1000.0
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    A_all = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    U_full = de.unit_potential(V, F, pts_m)
    U_R = de.unit_potential(Vr, Fr, pts_m)
    om = 2.0 * np.pi / (PERIOD * 3600.0)
    U_rot = -0.5 * om ** 2 * (pts_m[:, 0] ** 2 + pts_m[:, 1] ** 2)

    Am = A_all[keep]
    w = Am / Am.sum()
    Uf, Ur, Uo = U_full[keep], U_R[keep], U_rot[keep]

    def disp(rho_R):
        rho_rest = (M_tot - rho_R * V_R * 1e9) / ((V_tot - V_R) * 1e9)
        U = rho_rest * Uf + (rho_R - rho_rest) * Ur + Uo
        Ub = float((w * U).sum())
        return float(np.sqrt((w * (U - Ub) ** 2).sum())) / abs(Ub), rho_rest

    grid = np.arange(scan[0], scan[1] + 0.5 * scan[2], scan[2])
    vals, rests = zip(*(disp(g) for g in grid))
    vals = np.array(vals)
    i = int(np.nanargmin(vals))
    uniform, _ = disp(M_tot / (V_tot * 1e9))
    return dict(
        rho_grid=[float(g) for g in grid],
        dispersion=[float(v) for v in vals],
        rho_R=float(grid[i]), rho_rest=float(rests[i]),
        contrast=float(grid[i] / rests[i]),
        dispersion_min=float(vals[i]), dispersion_uniform=float(uniform),
        improvement_pct=float(100 * (uniform - vals[i]) / uniform),
        interior_minimum=bool(0 < i < len(grid) - 1),
        V_R_km3=float(V_R), V_tot_km3=float(V_tot),
        excess_mass_pct=float(100 * (grid[i] - rests[i]) * V_R * 1e9 / M_tot),
        facets_used=int(keep.sum()))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    t0 = time.time()
    V, F = load_obj(sys.argv[1])
    print(f"{len(F)} facets, {len(V)} vertices")

    buried, groups = find_buried(V, F)
    keep = ~buried
    A = de.face_areas(V, F)
    print(f"{len(groups)} components; {buried.sum()} facets buried, "
          f"{100 * A[buried].sum() / A.sum():.3f} per cent of the area")

    V_tot = de.closed_volume(V, F)
    out = {"plane_km": PLANE, "period_h": PERIOD, "scan": SCAN,
           "buried_facets": int(buried.sum()),
           "buried_area_pct": float(100 * A[buried].sum() / A.sum()),
           "volume_km3": float(V_tot), "curves": {}}

    print(f"\n{'bulk':>6} {'rho_R':>7} {'rho_rest':>9} {'contrast':>9} "
          f"{'improve':>9} {'interior':>9}")
    for bulk in DENSITIES:
        M_tot = bulk * V_tot * 1e9
        c = curve(V, F, keep, M_tot, PLANE, SCAN)
        c["assumed_bulk"] = bulk
        c["M_total_kg"] = M_tot
        out["curves"][f"{bulk:.0f}"] = c
        print(f"{bulk:>6.0f} {c['rho_R']:>7.0f} {c['rho_rest']:>9.2f} "
              f"{c['contrast']:>9.4f} {c['improvement_pct']:>8.3f}% "
              f"{str(c['interior_minimum']):>9}")

    json.dump(out, open("arrokoth_scan_clean.json", "w"), indent=1)
    print(f"\nwritten to arrokoth_scan_clean.json  "
          f"[{(time.time() - t0) / 60:.1f} min]")
    print("The contrasts above should reproduce 1.0659, 0.9999, 0.9293 and "
          "0.9159;\nif they do not, the two runs differ and the difference "
          "must be found before\nthe curve is used in Figure 3.")


if __name__ == "__main__":
    main()
