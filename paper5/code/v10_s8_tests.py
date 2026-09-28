#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The two measurements Section 8 still needs from the shape models.

    python v10_s8_tests.py --itokawa "Itokawa Hayabusa 50k poly.obj" \
                           --eros    "Eros Gaskell 50k poly.obj" \
                           --bennu   "Bennu_v20_200k.obj" \
                           --cg      "cg_mspcd_shap2_001m_cart.obj" \
                           --arrokoth "arrokoth_porter_2024_v01.obj"

Writes v10_s8_tests.json.

1. THE OMEGA = 0 CONTROL   -> Section 8.2
   Section 8.2 argues that holding the measured mass fixed makes the inversion
   better conditioned than varying it, because the rotational term enters the
   objective but not the mass constraint. The direct test is to repeat the
   Itokawa reference inversion with the rotational potential set identically
   to zero. If the recovered contrast is materially unchanged, the argument is
   confirmed in the output rather than only in the derivation. Nothing is
   recomputed except the sum: the gravity evaluation is shared between the two
   runs, so the control costs seconds.

2. THE EQUATORIAL COLUMN   -> Table 8.1
   The table compares the rotational and gravitational terms body by body
   using a maximum semi-axis perpendicular to the spin axis taken from
   published axis ratios. Two entries are missing and the two that are
   present were not measured on the meshes actually used. This computes, for
   each model, the volume-equivalent radius and the largest distance from the
   spin axis reached by any vertex, and the ratio of the centrifugal to the
   mean gravitational acceleration at that radius.

The third outstanding item, the 49,152-facet row of Table 8.3, comes from
test_ellipsoid.py: add 49152 to its TARGETS list and rerun. It needs no shape
model and takes about ten minutes.
"""

import argparse
import json
import sys

import numpy as np

import density_estimate as de
from surface_slope_lib import load_obj, outward_normals

G = 6.67430e-11
REF = dict(mass=3.58e10, period=12.1324, x=0.1608, scan=(300, 4000, 25))


def invert(V, F, x, mass, period, scan, with_rotation=True):
    V_tot = de.closed_volume(V, F)
    Vr, Fr = de.clip_halfspace(V, F, 0, x, True)
    V_R = de.closed_volume(Vr, Fr)
    nf = outward_normals(V, F)
    C = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    ext = float(np.linalg.norm(V.max(0) - V.min(0)))
    pts = (C - 1e-4 * ext * nf) * 1000.0
    A = de.face_areas(V, F)
    w = A / A.sum()
    Uf = de.unit_potential(V, F, pts)
    Ur = de.unit_potential(Vr, Fr, pts)
    if with_rotation:
        om = 2 * np.pi / (period * 3600.0)
        Uo = -0.5 * om ** 2 * (pts[:, 0] ** 2 + pts[:, 1] ** 2)
    else:
        Uo = np.zeros(len(pts))

    def disp(rR):
        rr = (mass - rR * V_R * 1e9) / ((V_tot - V_R) * 1e9)
        U = rr * Uf + (rR - rr) * Ur + Uo
        Ub = float((w * U).sum())
        return float(np.sqrt((w * (U - Ub) ** 2).sum())) / abs(Ub), rr

    grid = np.arange(scan[0], scan[1] + 1, scan[2])
    vals = np.array([disp(g)[0] for g in grid])
    i = int(np.nanargmin(vals))
    dmin, rr = disp(grid[i])
    duni, _ = disp(mass / (V_tot * 1e9))
    return dict(rho_R=float(grid[i]), rho_rest=float(rr),
                contrast=float(grid[i]) / rr, dispersion_uniform=duni,
                dispersion_min=dmin,
                improvement=100 * (duni - dmin) / duni,
                interior=bool(0 < i < len(grid) - 1))


def geometry(V, F, mass, period):
    """Volume-equivalent radius, maximum equatorial radius, and the ratio of
    the centrifugal to the mean gravitational acceleration there."""
    vol = abs(de.closed_volume(V, F)) * 1e9
    Rv = (3 * vol / (4 * np.pi)) ** (1 / 3)
    req = float(np.max(np.hypot(V[:, 0], V[:, 1]))) * 1000.0
    g_mean = G * mass / Rv ** 2
    om = 2 * np.pi / (period * 3600.0)
    return dict(R_v_m=Rv, R_eq_m=req, ratio_eq_over_Rv=req / Rv,
                g_at_Rv=g_mean, a_cent_at_Req=om ** 2 * req,
                a_over_g=om ** 2 * req / g_mean)


BODIES = {  # mass in kg, period in hours; Arrokoth at the 400 kg/m3 assumption
    "itokawa": (3.58e10, 12.1324),
    "eros": (6.687e15, 5.27),
    "bennu": (7.330e10, 4.296061),
    "cg": (9.982e12, 12.4043),
    "arrokoth": (1.6495e15, 15.92),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for b in BODIES:
        ap.add_argument(f"--{b}")
    ap.add_argument("--record", default="v10_s8_tests.json")
    a = ap.parse_args()
    out = {}

    if a.itokawa:
        print("1. OMEGA = 0 CONTROL, Itokawa at the reference plane")
        V, F = load_obj(a.itokawa)
        on = invert(V, F, REF["x"], REF["mass"], REF["period"], REF["scan"],
                    True)
        off = invert(V, F, REF["x"], REF["mass"], REF["period"], REF["scan"],
                     False)
        out["omega_control"] = dict(with_rotation=on, without_rotation=off)
        for lab, r in (("with rotation   ", on), ("without rotation", off)):
            print(f"   {lab}: rho_R {r['rho_R']:.0f}  rho_rest "
                  f"{r['rho_rest']:.1f}  contrast {r['contrast']:.4f}  "
                  f"improvement {r['improvement']:.3f}%")
        dc = 100 * (off["contrast"] / on["contrast"] - 1)
        print(f"   -> contrast changes by {dc:+.2f}%, rho_R by "
              f"{off['rho_R'] - on['rho_R']:+.0f} kg/m3\n")

    print("2. EQUATORIAL GEOMETRY")
    print(f"   {'body':<10}{'R_v (m)':>12}{'R_eq (m)':>12}{'R_eq/R_v':>10}"
          f"{'a_c/g':>9}")
    out["geometry"] = {}
    for b, (mass, period) in BODIES.items():
        path = getattr(a, b)
        if not path:
            continue
        V, F = load_obj(path)
        g = geometry(V, F, mass, period)
        out["geometry"][b] = g
        print(f"   {b:<10}{g['R_v_m']:>12.1f}{g['R_eq_m']:>12.1f}"
              f"{g['ratio_eq_over_Rv']:>10.3f}{g['a_over_g']:>9.4f}")

    json.dump(out, open(a.record, "w"), indent=1)
    print(f"\nwritten to {a.record}")


if __name__ == "__main__":
    sys.exit(main())
