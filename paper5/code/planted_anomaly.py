#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planted-anomaly recovery: does the inversion return a contrast it was given?

    python planted_anomaly.py                 the full experiment
    python planted_anomaly.py --quick         two contrasts, coarse mesh
    python planted_anomaly.py --selftest      check the relaxation machinery

Why the obvious version of this test is empty
---------------------------------------------
The natural experiment is to take a synthetic body, assign it a known interior
contrast, compute its surface potential, and ask the inversion to recover the
contrast. That tests nothing. The objective function

    sigma_U(rho_R) = sd(U(rho_R)) / |mean(U(rho_R))|

is built from the SHAPE and the trial density. It never reads a potential that
was computed beforehand. Planting a contrast in a body and leaving its shape
alone changes nothing the minimization can see, so the recovered value is
whatever the shape alone would have given, and the comparison is meaningless.

What the method actually assumes
--------------------------------
Richardson and Bowling's premise is that the surface has relaxed toward an
equipotential of the true interior. A body with a dense head and a relaxed
surface is therefore not the same shape as a uniform body: the surface has
moved. That is the content of the assumption, and it is what makes the
inversion possible at all.

So the honest ground-truth test is to build the body the assumption describes:

    1. choose a target contrast, say rho_R / rho_rest = 1.5
    2. start from a plausible shape
    3. relax the surface toward an equipotential of THAT two-region interior,
       by moving each vertex along its normal in proportion to the local
       departure of the geopotential from the mean
    4. iterate until the surface is as close to an equipotential as it will get
    5. hand the resulting shape to the inversion, which knows only the shape
       and the total mass, and see what contrast it returns

If the method works, step 5 returns the contrast chosen in step 1. If it does
not, the failure is informative: it means the objective can be minimised by a
density other than the one that generated the surface, which is a statement
about identifiability rather than about numerics.

This also yields the detection threshold the paper says it lacks. Run the
sequence for a ladder of contrasts down to 1.0 and find where the recovered
value stops tracking the planted one.

Caveats
-------
The relaxation in step 3 is a crude surrogate for a geophysical process. It
moves the surface toward an equipotential, which is what the premise asserts
the real process achieves, but it says nothing about whether regolith would
actually flow that way. The experiment tests the inverse problem, not the
geophysics.

Relaxation does not converge to a perfect equipotential on a two-region body:
the surface is constrained to remain a closed star-shaped mesh, and no such
surface is exactly equipotential in general. The residual is reported, and the
recovery should be read against it.
"""

import argparse
import json
import sys
import time

import numpy as np

import density_estimate as de
from surface_slope_lib import outward_normals


G = 6.67430e-11


# ---------------------------------------------------------------- geometry
def start_body(n_theta=120, n_phi=60, a=0.20, b=0.13, c=0.11, neck=0.72):
    """A mildly bilobed starting shape, star-shaped about the origin.

    Star-shaped means every ray from the origin meets the surface once, which
    keeps the relaxation step well defined: a vertex can be moved along its own
    radius without the surface folding through itself.
    """
    th = np.linspace(0, np.pi, n_phi + 2)[1:-1]
    ph = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    # radius modulated along x to make two lobes joined by a neck
    ct = np.cos(T)
    waist = 1.0 - (1.0 - neck) * np.exp(-((ct / 0.28) ** 2))
    r = waist * np.sqrt((a * np.sin(T) * np.cos(P)) ** 2
                        + (b * np.sin(T) * np.sin(P)) ** 2
                        + (c * ct) ** 2) / np.sqrt(
        np.sin(T) ** 2 * np.cos(P) ** 2 + np.sin(T) ** 2 * np.sin(P) ** 2
        + ct ** 2)
    # rebuild as an explicit direction times radius so the mesh is star-shaped
    D = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P),
                  np.cos(T) * np.ones_like(P)], axis=-1)
    D = D / np.linalg.norm(D, axis=-1)[..., None]
    scale = 1.0 / np.sqrt((D[..., 0] / a) ** 2 + (D[..., 1] / b) ** 2
                          + (D[..., 2] / c) ** 2)
    R = waist * scale
    V = (D * R[..., None]).reshape(-1, 3)
    # x is the long axis: rotate so the lobes lie along x
    V = V[:, [2, 0, 1]]

    F = []
    for i in range(n_phi - 1):
        for j in range(n_theta):
            j2 = (j + 1) % n_theta
            a0 = i * n_theta + j
            a1 = i * n_theta + j2
            b0 = (i + 1) * n_theta + j
            b1 = (i + 1) * n_theta + j2
            F += [[a0, a1, b1], [a0, b1, b0]]
    # caps
    top = len(V)
    V = np.vstack([V, [[0, 0, 0]]])
    V[top] = np.array([1.02 * V[:top, 0].max(), 0, 0])
    bot = len(V)
    V = np.vstack([V, [[1.02 * V[:top, 0].min(), 0, 0]]])
    for j in range(n_theta):
        j2 = (j + 1) % n_theta
        F.append([top, j2, j])
        F.append([bot, (n_phi - 1) * n_theta + j,
                  (n_phi - 1) * n_theta + j2])
    F = np.array(F, int)
    if de.closed_volume(V, F) < 0:
        F = F[:, ::-1]
    return V, F


def geopotential(V, F, rho_R, rho_rest, x_split, period_h):
    """Surface geopotential of the two-region body, at facet centroids."""
    nf = outward_normals(V, F)
    C = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    ext = float(np.linalg.norm(V.max(0) - V.min(0)))
    pts_km = C - 1e-4 * ext * nf
    pts_m = pts_km * 1000.0

    U_full = de.unit_potential(V, F, pts_m)
    Vr, Fr = de.clip_halfspace(V, F, 0, x_split, True)
    U_R = de.unit_potential(Vr, Fr, pts_m)

    U = rho_rest * U_full + (rho_R - rho_rest) * U_R
    om = 2.0 * np.pi / (period_h * 3600.0)
    U = U - 0.5 * om ** 2 * (pts_m[:, 0] ** 2 + pts_m[:, 1] ** 2)
    return U, C


def relax(V, F, rho_R, rho_rest, x_split, period_h, n_iter=40, gain=0.35,
          verbose=False):
    """Move the surface toward an equipotential of the given interior.

    Each vertex is displaced along its own radius by an amount proportional to
    the local departure of the geopotential from the area-weighted mean, scaled
    so that the total volume is preserved. Radial motion keeps the mesh
    star-shaped and cannot fold it.
    """
    V = V.copy()
    V0_vol = abs(de.closed_volume(V, F))
    hist = []
    for it in range(n_iter):
        U, C = geopotential(V, F, rho_R, rho_rest, x_split, period_h)
        p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        w = A / A.sum()
        Ubar = float((w * U).sum())
        disp = float(np.sqrt((w * (U - Ubar) ** 2).sum())) / abs(Ubar)
        hist.append(disp)
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            print(f"      iter {it:>3}  normalised dispersion {disp:.6f}")
        if it and abs(hist[-2] - disp) < 1e-9:
            break

        # facet residual -> vertex residual, area weighted
        res = (U - Ubar) / abs(Ubar)
        vres = np.zeros(len(V))
        vw = np.zeros(len(V))
        for k in range(3):
            np.add.at(vres, F[:, k], res * A)
            np.add.at(vw, F[:, k], A)
        vres /= np.maximum(vw, 1e-300)

        # a point where the geopotential is high sits too low: move it out.
        # The sign convention: U is negative, so a LESS negative U means the
        # point is further from the mass, and it should move in.
        r = np.linalg.norm(V, axis=1)
        step = -gain * vres * r
        V = V * (1.0 + (step / np.maximum(r, 1e-300))[:, None])

        # restore the volume, so that comparisons are at fixed size
        V *= (V0_vol / abs(de.closed_volume(V, F))) ** (1.0 / 3.0)
    return V, hist


# ------------------------------------------------------------------- driver
def one_case(contrast, frac=0.35, period_h=12.13, bulk=2000.0, n_theta=120,
             n_phi=60, n_iter=40, verbose=False):
    V, F = start_body(n_theta=n_theta, n_phi=n_phi)
    V_tot = abs(de.closed_volume(V, F))
    M_tot = bulk * V_tot * 1e9

    # place the plane so the region holds `frac` of the volume
    lo, hi = float(V[:, 0].min()), float(V[:, 0].max())
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        Vr, Fr = de.clip_halfspace(V, F, 0, mid, True)
        f = abs(de.closed_volume(Vr, Fr)) / V_tot
        if f > frac:
            lo = mid
        else:
            hi = mid
    x_split = 0.5 * (lo + hi)
    V_R = abs(de.closed_volume(*de.clip_halfspace(V, F, 0, x_split, True)))

    # the planted interior, at the chosen contrast and the fixed total mass
    rho_rest = M_tot / ((contrast - 1.0) * V_R * 1e9 + V_tot * 1e9)
    rho_R = contrast * rho_rest

    if verbose:
        print(f"    planted: rho_R = {rho_R:.1f}, rho_rest = {rho_rest:.1f}, "
              f"contrast = {contrast:.3f}, region {100*V_R/V_tot:.1f}%")

    Vrel, hist = relax(V, F, rho_R, rho_rest, x_split, period_h,
                       n_iter=n_iter, verbose=verbose)

    # the inversion sees only the relaxed shape and the total mass
    res = invert(Vrel, F, M_tot, period_h, x_split, bulk)
    return {
        "contrast_planted": contrast,
        "rho_R_planted": rho_R, "rho_rest_planted": rho_rest,
        "region_fraction": float(V_R / V_tot),
        "x_split": float(x_split),
        "dispersion_before": hist[0], "dispersion_after": hist[-1],
        "relax_iterations": len(hist),
        **res,
    }


def invert(V, F, M_tot, period_h, x_split, bulk):
    """Scan rho_R for the minimum of the normalised dispersion."""
    V_tot = abs(de.closed_volume(V, F))
    Vr, Fr = de.clip_halfspace(V, F, 0, x_split, True)
    V_R = abs(de.closed_volume(Vr, Fr))

    nf = outward_normals(V, F)
    C = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    ext = float(np.linalg.norm(V.max(0) - V.min(0)))
    pts_m = (C - 1e-4 * ext * nf) * 1000.0
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    w = A / A.sum()

    U_full = de.unit_potential(V, F, pts_m)
    U_R = de.unit_potential(Vr, Fr, pts_m)
    om = 2.0 * np.pi / (period_h * 3600.0)
    U_rot = -0.5 * om ** 2 * (pts_m[:, 0] ** 2 + pts_m[:, 1] ** 2)

    def disp(rho_R):
        rho_rest = (M_tot - rho_R * V_R * 1e9) / ((V_tot - V_R) * 1e9)
        U = rho_rest * U_full + (rho_R - rho_rest) * U_R + U_rot
        Ub = float((w * U).sum())
        return float(np.sqrt((w * (U - Ub) ** 2).sum())) / abs(Ub), rho_rest

    grid = np.arange(0.2 * bulk, 3.0 * bulk + 1, 0.005 * bulk)
    vals = np.array([disp(g)[0] for g in grid])
    i = int(np.nanargmin(vals))
    rho_R = float(grid[i])
    d_min, rho_rest = disp(rho_R)
    d_uni, _ = disp(M_tot / (V_tot * 1e9))
    return {
        "rho_R_recovered": rho_R,
        "rho_rest_recovered": float(rho_rest),
        "contrast_recovered": float(rho_R / rho_rest),
        "improvement_pct": float(100 * (d_uni - d_min) / d_uni),
        "interior_minimum": bool(0 < i < len(grid) - 1),
        "dispersion_uniform": float(d_uni),
        "dispersion_min": float(d_min),
    }


def selftest():
    print("SELF-TEST: does the relaxation reduce the dispersion?")
    V, F = start_body(n_theta=60, n_phi=30)
    ok, _ = de.check_manifold(F)
    print(f"  starting mesh: {len(F)} facets, closed manifold {ok}, "
          f"volume {abs(de.closed_volume(V, F)):.6f} km^3")
    if not ok:
        print("  FAIL: the starting mesh is not a closed manifold")
        return 1
    V_tot = abs(de.closed_volume(V, F))
    M = 2000.0 * V_tot * 1e9
    Vr, Fr = de.clip_halfspace(V, F, 0, 0.0, True)
    fR = abs(de.closed_volume(Vr, Fr)) / V_tot
    rest = M / (0.5 * fR * V_tot * 1e9 + V_tot * 1e9)
    Vrel, hist = relax(V, F, 1.5 * rest, rest, 0.0, 12.13, n_iter=20,
                       verbose=True)
    ok2, _ = de.check_manifold(F)
    print(f"  dispersion {hist[0]:.6f} -> {hist[-1]:.6f} "
          f"({100*(hist[-1]/hist[0]-1):+.1f}%)")
    print(f"  volume preserved to "
          f"{100*abs(abs(de.closed_volume(Vrel, F))/V_tot - 1):.4f}%")
    good = hist[-1] < hist[0] and ok2
    print(f"  RESULT: {'PASS' if good else 'FAIL'}")
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default="planted_anomaly.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    contrasts = [1.0, 1.25, 1.75] if a.quick else \
                [1.0, 1.05, 1.10, 1.25, 1.50, 1.75, 2.00]
    nt, np_, ni = (60, 30, 20) if a.quick else (120, 60, 40)

    print(f"PLANTED-ANOMALY RECOVERY  ({len(contrasts)} contrasts)")
    print(f"Each case builds a body whose surface is relaxed toward an "
          f"equipotential\nof the planted interior, then inverts it knowing "
          f"only the shape and the mass.\n")
    out, t0 = [], time.time()
    for c in contrasts:
        print(f"  contrast {c:.2f} ...")
        r = one_case(c, n_theta=nt, n_phi=np_, n_iter=ni, verbose=a.verbose)
        out.append(r)
        print(f"    planted {c:.3f} -> recovered "
              f"{r['contrast_recovered']:.3f}   "
              f"improvement {r['improvement_pct']:.2f}%   "
              f"surface dispersion {r['dispersion_before']:.5f} -> "
              f"{r['dispersion_after']:.5f}")
        json.dump(out, open(a.record, "w"), indent=1)

    print(f"\n{'='*74}\nRECOVERY CURVE\n{'='*74}")
    print(f"{'planted':>9} {'recovered':>11} {'error':>9} "
          f"{'improvement':>12} {'residual disp':>14}")
    for r in out:
        e = r["contrast_recovered"] - r["contrast_planted"]
        print(f"{r['contrast_planted']:>9.3f} "
              f"{r['contrast_recovered']:>11.3f} {e:>+9.3f} "
              f"{r['improvement_pct']:>11.2f}% "
              f"{r['dispersion_after']:>14.6f}")

    null = [r for r in out if abs(r["contrast_planted"] - 1.0) < 1e-9]
    if null:
        base = null[0]["improvement_pct"]
        print(f"\n  uniform control: improvement {base:.3f}% and recovered "
              f"contrast {null[0]['contrast_recovered']:.3f}")
        above = [r for r in out if r["improvement_pct"] > 3 * max(base, 1e-6)]
        if above:
            lo = min(r["contrast_planted"] for r in above
                     if r["contrast_planted"] > 1.0)
            print(f"  lowest planted contrast whose improvement exceeds three "
                  f"times the control: {lo:.2f}")
        else:
            print("  no planted contrast exceeds three times the control; the "
                  "threshold is above the range tested")

    print(f"\nwritten to {a.record}   [{(time.time()-t0)/60:.1f} min]")
    print("\nRead the recovery column against the residual dispersion. A body "
          "whose\nsurface did not relax close to an equipotential is not a "
          "fair test of the\npremise, and its recovery should not be counted "
          "against the method.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
