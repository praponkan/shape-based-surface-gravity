#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does tilting the spin axis to the recovered principal axis change the answer?

    python v9_axis_test.py "Itokawa Hayabusa 50k poly.obj"

Section 2.1 of the V9 draft says the spin axis is held at the mesh z-axis and
promises to report "the change in the dispersion that results from recomputing
it self-consistently". Section 3 does not contain that result. This supplies
it.

The recovered two-region interior puts the axis of maximum moment 1.87 degrees
from the mesh z-axis, against 0.77 degrees for the uniform body. Here the
inversion is repeated with the rotational potential taken about that axis,
through the recovered centre of mass, and the result compared with the
reference run. Needs v9_section2_tests.py in the same directory.
"""

import json
import sys

import numpy as np

import density_estimate as de
from surface_slope_lib import load_obj, outward_normals
from v9_section2_tests import inertia

M, PERIOD, X, SCAN = 3.58e10, 12.1324, 0.1608, (300, 4000, 25)


def principal(V, F, Vr, Fr, rho_rest, rho_R):
    If, vf = inertia(V, F)
    Ir, vr = inertia(Vr, Fr)
    cf, cr = de.closed_centroid(V, F), de.closed_centroid(Vr, Fr)
    mf, mr = rho_rest * vf, (rho_R - rho_rest) * vr
    com = (mf * cf + mr * cr) / (mf + mr)
    I0 = rho_rest * If + (rho_R - rho_rest) * Ir
    Ic = I0 - (mf + mr) * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    w, v = np.linalg.eigh(Ic)
    ax = v[:, np.argmax(w)]
    return (ax if ax[2] > 0 else -ax), com


def invert(V, F, Vr, Fr, axis, centre):
    V_tot, V_R = de.closed_volume(V, F), de.closed_volume(Vr, Fr)
    nf = outward_normals(V, F)
    C = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    ext = float(np.linalg.norm(V.max(0) - V.min(0)))
    pts = (C - 1e-4 * ext * nf) * 1000.0
    A = de.face_areas(V, F)
    w = A / A.sum()
    Uf, Ur = de.unit_potential(V, F, pts), de.unit_potential(Vr, Fr, pts)
    om = 2 * np.pi / (PERIOD * 3600.0)
    r = pts - centre * 1000.0
    perp = r - np.outer(r @ axis, axis)
    Uo = -0.5 * om ** 2 * (perp ** 2).sum(1)

    def disp(rR):
        rr = (M - rR * V_R * 1e9) / ((V_tot - V_R) * 1e9)
        U = rr * Uf + (rR - rr) * Ur + Uo
        Ub = float((w * U).sum())
        return float(np.sqrt((w * (U - Ub) ** 2).sum())) / abs(Ub), rr

    grid = np.arange(SCAN[0], SCAN[1] + 1, SCAN[2])
    vals = np.array([disp(g)[0] for g in grid])
    i = int(np.nanargmin(vals))
    dmin, rr = disp(grid[i])
    duni, _ = disp(M / (V_tot * 1e9))
    return dict(rho_R=float(grid[i]), rho_rest=float(rr),
                contrast=float(grid[i] / rr), dispersion=dmin,
                improvement=100 * (duni - dmin) / duni,
                interior=bool(0 < i < len(grid) - 1))


def main():
    V, F = load_obj(sys.argv[1])
    Vr, Fr = de.clip_halfspace(V, F, 0, X, True)
    z, o = np.array([0.0, 0.0, 1.0]), np.zeros(3)

    ref = invert(V, F, Vr, Fr, z, o)
    ax, com = principal(V, F, Vr, Fr, ref["rho_rest"], ref["rho_R"])
    tilt = np.degrees(np.arccos(ax[2]))
    new = invert(V, F, Vr, Fr, ax, com)

    print(f"reference, axis z through origin : contrast {ref['contrast']:.4f}"
          f"  improvement {ref['improvement']:.3f}%  dispersion "
          f"{ref['dispersion']:.6f}")
    print(f"principal axis ({tilt:.2f} deg) via COM: contrast "
          f"{new['contrast']:.4f}  improvement {new['improvement']:.3f}%  "
          f"dispersion {new['dispersion']:.6f}")
    dc = 100 * (new["contrast"] / ref["contrast"] - 1)
    dd = 100 * (new["dispersion"] / ref["dispersion"] - 1)
    print(f"\nchange: contrast {dc:+.3f}%, dispersion at minimum {dd:+.3f}%, "
          f"improvement {new['improvement'] - ref['improvement']:+.3f} pts")
    json.dump(dict(tilt_deg=tilt, reference=ref, tilted=new,
                   contrast_change_pct=dc, dispersion_change_pct=dd),
              open("v9_axis_test.json", "w"), indent=1)
    print("written to v9_axis_test.json")


if __name__ == "__main__":
    main()
