#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test A: which decimator preserves the surface fields, judged against an
analytic reference rather than against a finer mesh.

    python test_ellipsoid.py

Why this test
-------------
The comparison of decimators in check_paper4.py concluded that vertex
clustering biases the mean surface slope downward on coarse meshes while
quadric edge collapse does not. That conclusion rested on taking the finest
mesh as the truth, which is reasonable but is not proof: if both decimators
were biased in the same direction, or if the quadric implementation were
wrong, the comparison would not reveal it.

A homogeneous ellipsoid removes the assumption. Its interior and surface
gravity have an exact closed form (MacMillan 1930; Kellogg 1929):

    g_i = -2 pi G rho a b c A_i x_i ,
    A_i = integral_0^inf du / [ (a_i^2 + u) sqrt((a^2+u)(b^2+u)(c^2+u)) ]

which reduces to GM/R^2 in the sphere limit, verified here to machine
precision. The slope at any surface point follows from the analytic gravity
and the analytic outward normal of the ellipsoid, with no mesh involved, and
the area-weighted mean slope is then a surface integral evaluated by
quadrature on the exact surface.

So there is a reference that owes nothing to the polyhedral code, to either
decimator, or to any mesh. The question becomes simple: at a given face count,
which decimation reproduces it better?

What is tested
--------------
    1. the analytic formula against the sphere limit          (formula check)
    2. the polyhedral library against the formula, fine mesh  (library check)
    3. mean slope and dispersion, both decimators, versus the analytic value
    4. the same for a rotating ellipsoid, where the centrifugal term matters

A note on what a "correct" answer means here. The analytic slope is a property
of the ellipsoid; a mesh approximates the ellipsoid, so even an undecimated
mesh carries a discretisation error. What the test measures is whether
decimation moves the answer further from the truth than the discretisation
alone already does, and by how much, for each method.
"""

import argparse
import json
import sys
import time

import numpy as np
from scipy import integrate

import density_estimate as de
from surface_slope_lib import outward_normals, decimate_cluster

try:
    import polyhedral_gravity as pg
except ImportError:
    sys.exit("install the gravity library first:\n"
             "    pip install polyhedral-gravity --break-system-packages")

G = 6.67430e-11

# semi-axes in km, and the rotation periods to test
AXES = [(1.0, 0.7, 0.5), (1.0, 0.9, 0.8), (1.0, 0.5, 0.35)]
# A 1 km body at 2000 kg/m^3 sheds material below a period of about 2.3 h, so
# a fast spin would put the mean slope above 90 degrees and test nothing
# realistic. At 12 h the centrifugal term is 3.8% of gravity at the equator,
# close to Itokawa; at 24 h it is 0.9%.
PERIODS = [None, 12.0, 24.0]
RHO = 2000.0                  # kg/m^3
TARGETS = [2000, 4000, 8000, 16000, 32000]
SUBDIV = 6                    # icosphere subdivision for the source mesh
# Subdivision 6 gives 20,480 facets, enough for every target above. The
# 49,152-facet row of the paper needs subdivision 7, which gives 81,920 facets
# and four times the work: use --subdiv 7 --targets 49152 to add that row on
# its own, and merge the result with the main record afterwards.


# ----------------------------------------------------------------- analytic
def shape_integral(a, b, c, i):
    """A_i for a homogeneous ellipsoid, dimension 1/length^3.

    Substituting u = a^2 v gives A_i(a,b,c) = a^-3 A_i(1, b/a, c/a), which
    keeps the integrand of order unity whatever the units of a, b, c. Without
    it, quad silently loses accuracy when the semi-axes are given in metres:
    the integral then runs to u of order 10^6 and the sphere check fails by a
    factor of several hundred, which is how this was found.
    """
    beta, gamma = b / a, c / a
    d2 = (1.0, beta ** 2, gamma ** 2)[i]
    f = lambda v: 1.0 / ((d2 + v) * np.sqrt((1.0 + v) * (beta ** 2 + v)
                                            * (gamma ** 2 + v)))
    val, err = integrate.quad(f, 0.0, np.inf, limit=400)
    return val / a ** 3


def shape_integral_0(a, b, c):
    """A_0, the constant term of the interior potential, dimension 1/length."""
    beta, gamma = b / a, c / a
    f = lambda v: 1.0 / np.sqrt((1.0 + v) * (beta ** 2 + v) * (gamma ** 2 + v))
    val, _ = integrate.quad(f, 0.0, np.inf, limit=400)
    return val / a


def analytic_gravity(pts_m, a_m, b_m, c_m, rho):
    """Gravity vector at points on or inside the ellipsoid, SI, pointing in."""
    A = np.array([shape_integral(a_m, b_m, c_m, i) for i in range(3)])
    k = 2.0 * np.pi * G * rho * a_m * b_m * c_m
    return -k * A[None, :] * pts_m


def analytic_potential(pts_m, a_m, b_m, c_m, rho):
    """Gravitational potential inside a homogeneous ellipsoid, SI, negative."""
    A = np.array([shape_integral(a_m, b_m, c_m, i) for i in range(3)])

    k = np.pi * G * rho * a_m * b_m * c_m
    quad = (A[None, :] * pts_m ** 2).sum(axis=1)
    return -k * (shape_integral_0(a_m, b_m, c_m) - quad)


def analytic_mean_slope(a, b, c, period_h, rho, n_theta=400, n_phi=800):
    """Area-weighted mean slope over the exact ellipsoid, by quadrature.

    Parametrised by the sphere angles and mapped to the ellipsoid; the area
    element carries the Jacobian, so the weighting is by true surface area.
    """
    a_m, b_m, c_m = a * 1000.0, b * 1000.0, c * 1000.0
    th = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    ph = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    T, P = np.meshgrid(th, ph, indexing="ij")
    x = a_m * np.sin(T) * np.cos(P)
    y = b_m * np.sin(T) * np.sin(P)
    z = c_m * np.cos(T)
    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # outward normal of the ellipsoid: gradient of x^2/a^2 + ... = 1
    nrm = np.column_stack([pts[:, 0] / a_m ** 2, pts[:, 1] / b_m ** 2,
                           pts[:, 2] / c_m ** 2])
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]

    g = analytic_gravity(pts, a_m, b_m, c_m, rho)
    if period_h:
        om = 2 * np.pi / (period_h * 3600.0)
        g = g + om ** 2 * np.column_stack([pts[:, 0], pts[:, 1],
                                           np.zeros(len(pts))])
    gn = np.linalg.norm(g, axis=1)
    cos = np.clip(np.einsum("ij,ij->i", -g, nrm) / np.maximum(gn, 1e-300),
                  -1.0, 1.0)
    slope = np.degrees(np.arccos(cos))

    # area element of the parametrised ellipsoid
    dT = np.pi / n_theta
    dP = 2 * np.pi / n_phi
    E = np.column_stack([a_m * np.cos(T).ravel() * np.cos(P).ravel(),
                         b_m * np.cos(T).ravel() * np.sin(P).ravel(),
                         -c_m * np.sin(T).ravel()])
    Fv = np.column_stack([-a_m * np.sin(T).ravel() * np.sin(P).ravel(),
                          b_m * np.sin(T).ravel() * np.cos(P).ravel(),
                          np.zeros(len(pts))])
    dA = np.linalg.norm(np.cross(E, Fv), axis=1) * dT * dP
    w = dA / dA.sum()

    U = analytic_potential(pts, a_m, b_m, c_m, rho)
    if period_h:
        om = 2 * np.pi / (period_h * 3600.0)
        U = U - 0.5 * om ** 2 * (pts[:, 0] ** 2 + pts[:, 1] ** 2)
    Ubar = float((w * U).sum())
    sd = float(np.sqrt((w * (U - Ubar) ** 2).sum()))

    return float((w * slope).sum()), sd / abs(Ubar)


# ------------------------------------------------------------------- meshed
def meshed_fields(V, F, rho, period_h):
    """Mean slope and dispersion from the polyhedral library, as Paper IV."""
    Vm = V * 1000.0
    nf = outward_normals(V, F)
    C = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3 * 1000.0
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    ext = float(np.linalg.norm((V.max(0) - V.min(0)) * 1000.0))
    pts = C - 1e-4 * ext * nf

    poly = pg.Polyhedron(
        polyhedral_source=(Vm.tolist(), F.tolist()), density=rho,
        normal_orientation=pg.NormalOrientation.OUTWARDS,
        integrity_check=pg.PolyhedronIntegrity.DISABLE)
    res = pg.evaluate(poly, pts.tolist(), parallel=True)
    U = -np.array([r[0] for r in res])
    g = np.array([r[1] for r in res])

    if period_h:
        om = 2 * np.pi / (period_h * 3600.0)
        g = g + om ** 2 * np.column_stack([pts[:, 0], pts[:, 1],
                                           np.zeros(len(pts))])
        U = U - 0.5 * om ** 2 * (pts[:, 0] ** 2 + pts[:, 1] ** 2)

    gn = np.linalg.norm(g, axis=1)
    cos = np.clip(np.einsum("ij,ij->i", -g, nf) / np.maximum(gn, 1e-300),
                  -1.0, 1.0)
    slope = np.degrees(np.arccos(cos))
    w = A / A.sum()
    Ubar = float((w * U).sum())
    sd = float(np.sqrt((w * (U - Ubar) ** 2).sum()))
    return float((w * slope).sum()), sd / abs(Ubar)


def ellipsoid_mesh(a, b, c, subdiv):
    V, F = de.icosphere(subdiv, 1.0)
    return V * np.array([a, b, c]), F


# -------------------------------------------------------------------- tests
def check_formula():
    print("1. THE ANALYTIC FORMULA, AGAINST THE SPHERE LIMIT")
    a = b = c = 1000.0
    Ax = shape_integral(a, b, c, 0)
    g_f = 2 * np.pi * G * RHO * a * b * c * Ax * a
    M = 4 / 3 * np.pi * a * b * c * RHO
    g_e = G * M / a ** 2
    d = abs(g_f / g_e - 1)
    print(f"   MacMillan  {g_f:.12e}")
    print(f"   GM/R^2     {g_e:.12e}")
    print(f"   difference {100*d:.3e} %   {'PASS' if d < 1e-10 else 'FAIL'}")
    return d < 1e-10


def check_library(a, b, c):
    print(f"\n2. THE POLYHEDRAL LIBRARY, AGAINST THE FORMULA "
          f"(a,b,c = {a},{b},{c})")
    V, F = ellipsoid_mesh(a, b, c, SUBDIV)
    Vm = V * 1000.0
    nf = outward_normals(V, F)
    C = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3 * 1000.0
    ext = float(np.linalg.norm((V.max(0) - V.min(0)) * 1000.0))
    pts = C - 1e-4 * ext * nf

    poly = pg.Polyhedron(
        polyhedral_source=(Vm.tolist(), F.tolist()), density=RHO,
        normal_orientation=pg.NormalOrientation.OUTWARDS,
        integrity_check=pg.PolyhedronIntegrity.DISABLE)
    res = pg.evaluate(poly, pts.tolist(), parallel=True)
    g_lib = np.array([r[1] for r in res])
    U_lib = -np.array([r[0] for r in res])

    g_an = analytic_gravity(pts, a * 1000, b * 1000, c * 1000, RHO)
    U_an = analytic_potential(pts, a * 1000, b * 1000, c * 1000, RHO)

    dg = np.abs(np.linalg.norm(g_lib, axis=1) / np.linalg.norm(g_an, axis=1)
                - 1)
    dU = np.abs(U_lib / U_an - 1)
    print(f"   facets {len(F)}")
    print(f"   |g|  median {100*np.median(dg):.4f}%  max {100*dg.max():.4f}%")
    print(f"   U    median {100*np.median(dU):.4f}%  max {100*dU.max():.4f}%")
    print(f"   (the residual is the polyhedron approximating a smooth "
          f"surface, not library error)")
    return float(np.median(dg)), float(np.median(dU))


def main():
    global TARGETS, SUBDIV
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subdiv", type=int, default=SUBDIV,
                    help="icosphere subdivision of the source mesh: 6 gives "
                         "20,480 facets, 7 gives 81,920")
    ap.add_argument("--targets", type=int, nargs="+", default=None,
                    help="facet counts to decimate to; each must be smaller "
                         "than the source mesh")
    ap.add_argument("--record", default="ellipsoid_test.json",
                    help="output file")
    a = ap.parse_args()
    SUBDIV = a.subdiv
    if a.targets:
        TARGETS = sorted(a.targets)
    print(f"source mesh: subdivision {SUBDIV}, "
          f"{20 * 4 ** (SUBDIV - 1)} facets;  targets {TARGETS}")

    t0 = time.time()
    ok = check_formula()
    if not ok:
        sys.exit("the analytic formula failed its own check; stop here")

    check_library(*AXES[0])

    print(f"\n3. DECIMATION AGAINST THE ANALYTIC VALUE")
    records = []
    for (a, b, c) in AXES:
        for P in PERIODS:
            ms_an, di_an = analytic_mean_slope(a, b, c, P, RHO)
            V, F = ellipsoid_mesh(a, b, c, SUBDIV)
            ms_full, di_full = meshed_fields(V, F, RHO, P)
            tag = f"{a}/{b}/{c}" + (f"  P={P}h" if P else "  no spin")
            print(f"\n   ellipsoid {tag}")
            print(f"   analytic          slope {ms_an:8.4f} deg   "
                  f"dispersion {di_an:.6f}")
            print(f"   full mesh {len(F):>6}  slope {ms_full:8.4f} deg "
                  f"({100*(ms_full/ms_an-1):+6.2f}%)   "
                  f"dispersion {di_full:.6f} ({100*(di_full/di_an-1):+6.2f}%)")
            print(f"   {'target':>7} {'method':<8} {'faces':>7} "
                  f"{'slope':>9} {'err':>8}   {'dispersion':>11} {'err':>8}")
            for t in TARGETS:
                if t >= len(F):
                    continue
                for kind in ("qem", "cluster"):
                    if kind == "qem":
                        from decimate_qem import decimate_qem
                        Vd, Fd = decimate_qem(V.copy(), F.copy(), t)
                    else:
                        Vd, Fd = decimate_cluster(V.copy(), F.copy(), t)
                    ms, di = meshed_fields(Vd, Fd, RHO, P)
                    e_s = 100 * (ms / ms_an - 1)
                    e_d = 100 * (di / di_an - 1)
                    print(f"   {t:>7} {kind:<8} {len(Fd):>7} {ms:>9.4f} "
                          f"{e_s:>+7.2f}%   {di:>11.6f} {e_d:>+7.2f}%")
                    records.append({
                        "axes": [a, b, c], "period_h": P, "target": t,
                        "method": kind, "faces": int(len(Fd)),
                        "slope_deg": ms, "dispersion": di,
                        "slope_analytic": ms_an, "dispersion_analytic": di_an,
                        "slope_error_pct": e_s, "dispersion_error_pct": e_d})

    json.dump(records, open(a.record, "w"), indent=1)

    print(f"\n{'='*88}\n4. SUMMARY: mean absolute error against the analytic "
          f"value\n{'='*88}")
    print(f"   {'target':>7} | {'slope, qem':>12} {'slope, cluster':>15} | "
          f"{'disp, qem':>11} {'disp, cluster':>14}")
    for t in TARGETS:
        sub = [r for r in records if r["target"] == t]
        if not sub:
            continue
        f = lambda k, m: np.mean([abs(r[k]) for r in sub
                                  if r["method"] == m]) if any(
            r["method"] == m for r in sub) else float("nan")
        print(f"   {t:>7} | {f('slope_error_pct','qem'):>11.2f}% "
              f"{f('slope_error_pct','cluster'):>14.2f}% | "
              f"{f('dispersion_error_pct','qem'):>10.2f}% "
              f"{f('dispersion_error_pct','cluster'):>13.2f}%")

    sq = np.mean([abs(r["slope_error_pct"]) for r in records
                  if r["method"] == "qem"])
    sc = np.mean([abs(r["slope_error_pct"]) for r in records
                  if r["method"] == "cluster"])
    dq = np.mean([abs(r["dispersion_error_pct"]) for r in records
                  if r["method"] == "qem"])
    dc = np.mean([abs(r["dispersion_error_pct"]) for r in records
                  if r["method"] == "cluster"])
    print(f"\n   overall mean absolute error")
    print(f"     slope       qem {sq:6.2f}%   cluster {sc:6.2f}%   "
          f"{'qem better' if sq < sc else 'cluster better'} "
          f"by a factor {max(sq,sc)/max(min(sq,sc),1e-9):.1f}")
    print(f"     dispersion  qem {dq:6.2f}%   cluster {dc:6.2f}%   "
          f"{'qem better' if dq < dc else 'cluster better'} "
          f"by a factor {max(dq,dc)/max(min(dq,dc),1e-9):.1f}")
    print(f"\n   Also note which quantity is the more robust overall: slope "
          f"errors average\n   {min(sq,sc):.2f}% at best against "
          f"{min(dq,dc):.2f}% for the dispersion.")
    print(f"\nwritten to {a.record}   [{(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    main()
