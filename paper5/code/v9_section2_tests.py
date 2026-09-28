#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The three numbers Section 2 of the V9 draft still needs.

    python v9_section2_tests.py --itokawa "Itokawa Hayabusa 50k poly.obj" \
                                --eros    "Eros Gaskell 50k poly.obj"

Writes v9_section2_tests.json and prints the values to insert.

1. OFFSET SENSITIVITY  -> [X] and [Y] in Section 2.1
   Field points sit inside each facet by 1e-4 of the bounding-box diagonal.
   The draft says every run was repeated with that offset scaled by 10 and by
   0.1. That was not done. This script does it for Itokawa at the reference
   plane and Eros at Himeros, the strong case and the best-resolved weak one,
   and reports the largest change in contrast (per cent) and in improvement
   (percentage points). If you want every body, add them to BODIES below.

2. PRINCIPAL-AXIS ROTATION  -> the empty \\ang{;;} in Section 2.1
   A two-region body has a different inertia tensor from the uniform one. The
   angle between the axis of maximum moment and the mesh z-axis is computed
   for the uniform Itokawa and for the recovered contrast, and the difference
   reported. The inertia of each polyhedron is summed over signed tetrahedra
   from the origin, which is exact for a closed mesh.

3. SLIVER TEST ON ITOKAWA  -> confirms the claim in Section 2.4
   The draft says that raising the clipping tolerance from 1e-10 to 1e-6
   left rho_head unchanged at every plane of the sweep and moved rho_rest by
   under 0.01 kg/m3. That was measured on 67P, not Itokawa. This runs the
   thirteen Itokawa planes at both tolerances and reports the largest change.

None of these needs the batch runner or the mesh cache. Expect about twenty
minutes in all, most of it the sliver test.
"""

import argparse
import json
import sys
import time

import numpy as np

import density_estimate as de
from surface_slope_lib import load_obj, outward_normals

G = 6.67430e-11

BODIES = {
    "Itokawa": dict(mass=3.58e10, period=12.1324, x=0.1608,
                    scan=(300, 4000, 25)),
    "Eros": dict(mass=6.687e15, period=5.27, x=4.4698, scan=(500, 5000, 25)),
}
SWEEP_X = [0.130, 0.140, 0.150, 0.160, 0.1608, 0.170, 0.180, 0.185, 0.190,
           0.195, 0.200, 0.220, 0.240]


# ------------------------------------------------------------------ inversion
def invert(V, F, M, period, x, scan, offset_frac=1e-4, tol_rel=None):
    V_tot = de.closed_volume(V, F)
    Vr, Fr = de.clip_halfspace(V, F, 0, x, True, tol_rel=tol_rel)
    V_R = de.closed_volume(Vr, Fr)
    nf = outward_normals(V, F)
    C = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    ext = float(np.linalg.norm(V.max(0) - V.min(0)))
    pts = (C - offset_frac * ext * nf) * 1000.0
    A = de.face_areas(V, F)
    w = A / A.sum()
    Uf = de.unit_potential(V, F, pts)
    Ur = de.unit_potential(Vr, Fr, pts)
    om = 2 * np.pi / (period * 3600.0)
    Uo = -0.5 * om ** 2 * (pts[:, 0] ** 2 + pts[:, 1] ** 2)

    def disp(rR):
        rr = (M - rR * V_R * 1e9) / ((V_tot - V_R) * 1e9)
        U = rr * Uf + (rR - rr) * Ur + Uo
        Ub = float((w * U).sum())
        return float(np.sqrt((w * (U - Ub) ** 2).sum())) / abs(Ub), rr

    lo, hi, st = scan
    grid = np.arange(lo, hi + 0.5 * st, st)
    vals = np.array([disp(g)[0] for g in grid])
    i = int(np.nanargmin(vals))
    rR = float(grid[i])
    dmin, rr = disp(rR)
    duni, _ = disp(M / (V_tot * 1e9))
    return dict(rho_R=rR, rho_rest=float(rr), contrast=rR / rr,
                improvement=100 * (duni - dmin) / duni,
                interior=bool(0 < i < len(grid) - 1), V_R=V_R, V_tot=V_tot)


# ------------------------------------------------------------------- inertia
def inertia(V, F):
    """Inertia tensor about the origin of a closed mesh at unit density, km^5.

    Sums signed tetrahedra from the origin using the covariance form, which is
    exact for any closed consistently oriented polyhedron.
    """
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    # second moment of each tetrahedron (0,a,b,c)
    s = a + b + c
    cov = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            cov[i, j] = np.sum(vol * (a[:, i] * a[:, j] + b[:, i] * b[:, j]
                                      + c[:, i] * c[:, j] + s[:, i] * s[:, j])
                               ) / 20.0
    return np.trace(cov) * np.eye(3) - cov, float(vol.sum())


def spin_axis_angle(V, F, Vr, Fr, rho_rest, rho_R):
    """Angle between the max-moment principal axis and the mesh z-axis, arcsec,
    for the two-region body, taken about its own centre of mass."""
    If, vf = inertia(V, F)
    Ir, vr = inertia(Vr, Fr)
    cf = de.closed_centroid(V, F)
    cr = de.closed_centroid(Vr, Fr)
    m_f = rho_rest * vf
    m_r = (rho_R - rho_rest) * vr
    I0 = rho_rest * If + (rho_R - rho_rest) * Ir
    com = (m_f * cf + m_r * cr) / (m_f + m_r)
    M = m_f + m_r
    Icom = I0 - M * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    w, v = np.linalg.eigh(Icom)
    ax = v[:, np.argmax(w)]
    ang = np.degrees(np.arccos(min(1.0, abs(ax[2])))) * 3600.0
    return ang, com


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--itokawa", required=True)
    ap.add_argument("--eros", required=True)
    ap.add_argument("--skip-sliver", action="store_true")
    ap.add_argument("--record", default="v9_section2_tests.json")
    a = ap.parse_args()
    out, t0 = {}, time.time()
    paths = {"Itokawa": a.itokawa, "Eros": a.eros}
    meshes = {k: load_obj(p) for k, p in paths.items()}

    # 1. offset sensitivity
    print("1. OFFSET SENSITIVITY  (offset x1, x10, x0.1)")
    dC, dI = [], []
    out["offset"] = {}
    for body, cfg in BODIES.items():
        V, F = meshes[body]
        res = {f: invert(V, F, cfg["mass"], cfg["period"], cfg["x"],
                         cfg["scan"], offset_frac=1e-4 * f)
               for f in (1.0, 10.0, 0.1)}
        base = res[1.0]
        for f in (10.0, 0.1):
            c = 100 * (res[f]["contrast"] / base["contrast"] - 1)
            i = res[f]["improvement"] - base["improvement"]
            dC.append(abs(c)); dI.append(abs(i))
            print(f"   {body:<8} x{f:<4}: contrast {res[f]['contrast']:.4f} "
                  f"({c:+.3f}%)  improvement {res[f]['improvement']:.3f}% "
                  f"({i:+.3f} pts)")
        out["offset"][body] = res
    print(f"   -> X = {max(dC):.2f}   Y = {max(dI):.3f}\n")
    out["X"], out["Y"] = max(dC), max(dI)

    # 2. principal-axis rotation, Itokawa at the recovered contrast
    print("2. PRINCIPAL-AXIS ROTATION, Itokawa")
    V, F = meshes["Itokawa"]
    cfg = BODIES["Itokawa"]
    r = out["offset"]["Itokawa"][1.0]
    Vr, Fr = de.clip_halfspace(V, F, 0, cfg["x"], True)
    bulk = cfg["mass"] / (r["V_tot"] * 1e9)
    a_uni, _ = spin_axis_angle(V, F, Vr, Fr, bulk, bulk)
    a_two, com = spin_axis_angle(V, F, Vr, Fr, r["rho_rest"], r["rho_R"])
    print(f"   uniform:    axis {a_uni:8.1f} arcsec from z")
    print(f"   two-region: axis {a_two:8.1f} arcsec from z, "
          f"COM at x = {1000*com[0]:.2f} m")
    print(f"   -> induced rotation {abs(a_two - a_uni):.1f} arcsec\n")
    out["axis"] = dict(uniform_arcsec=a_uni, two_region_arcsec=a_two,
                       rotation_arcsec=abs(a_two - a_uni),
                       com_x_m=1000 * com[0])

    # 3. sliver test across the Itokawa sweep
    if not a.skip_sliver:
        print("3. SLIVER TEST, Itokawa, tolerance 1e-10 vs 1e-6")
        drh, drr = 0.0, 0.0
        out["sliver"] = []
        for x in SWEEP_X:
            scan = (300, 4000, 25) if x < 0.185 else (300, 8000, 25)
            r1 = invert(V, F, cfg["mass"], cfg["period"], x, scan,
                        tol_rel=1e-10)
            r2 = invert(V, F, cfg["mass"], cfg["period"], x, scan,
                        tol_rel=1e-6)
            h = abs(r2["rho_R"] - r1["rho_R"])
            rr = abs(r2["rho_rest"] - r1["rho_rest"])
            drh, drr = max(drh, h), max(drr, rr)
            print(f"   x={x:.4f}: rho_head {r1['rho_R']:.0f}->{r2['rho_R']:.0f}"
                  f"  rho_rest diff {rr:.4f}")
            out["sliver"].append(dict(x=x, d_rho_head=h, d_rho_rest=rr))
        print(f"   -> largest change: rho_head {drh:.0f}, rho_rest {drr:.4f} "
              f"kg/m3")
        out["sliver_max"] = dict(rho_head=drh, rho_rest=drr)

    json.dump(out, open(a.record, "w"), indent=1, default=float)
    print(f"\nwritten to {a.record}   [{(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    sys.exit(main())
