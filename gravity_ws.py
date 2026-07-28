#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shape-based surface gravity: mean surface gravity via FULL Werner-Scheeres polyhedron gravity.
Uses the peer-reviewed 'polyhedral-gravity' library (Tsoulis formulation), which
implements the exact Werner-Scheeres (1997) field for a uniform polyhedron.

This is the PROJECT-STANDARD solver. It self-validates on a sphere before running,
and auto-calibrates the unit factor so results are returned as physical km/s^2.

------------------------------------------------------------------------------
INSTALL (one time):
    pip install numpy polyhedral-gravity
    # if your Python is "externally managed", add:  --break-system-packages

RUN:
    # self-test only:
    python gravity_ws.py
    # on a shape model (.obj in km, GM in km^3/s^2):
    python gravity_ws.py  "Eros Gaskell 50k poly.obj"  4.463e-4

GM reference values (km^3/s^2):
    Eros      4.463e-4        Itokawa   2.1e-9
    Bennu     4.892e-6        Phobos    7.072e-4
    (always double-check against the primary mission source you trust)
------------------------------------------------------------------------------
"""
import sys, math
import numpy as np

try:
    import polyhedral_gravity as pg
except ImportError:
    sys.exit("ERROR: install the library first:\n    pip install polyhedral-gravity --break-system-packages")

G_SI = 6.674e-11   # used only internally; cancels via calibration


# ---------------------------------------------------------------- mesh utils
def load_obj(path):
    V, F = [], []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for ln in fh:
            if ln.startswith('v '):
                p = ln.split(); V.append([float(p[1]), float(p[2]), float(p[3])])
            elif ln.startswith('f '):
                idx = [int(t.split('/')[0]) - 1 for t in ln.split()[1:]]
                for k in range(1, len(idx) - 1):
                    F.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(V, float), np.asarray(F, int)

def outward_normals(V, F):
    p0, p1, p2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    n = np.cross(p1 - p0, p2 - p0); n /= np.linalg.norm(n, axis=1)[:, None]
    ctr = V.mean(0); fc = (p0 + p1 + p2) / 3
    s = np.sign(np.einsum('ij,ij->i', n, fc - ctr)); s[s == 0] = 1
    return n * s[:, None]

def geometry(V, F):
    p0, p1, p2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    triA = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    A = triA.sum()
    Vvol = abs(np.sum(np.einsum('ij,ij->i', p0, np.cross(p1, p2))) / 6.0)
    R_A = math.sqrt(A / (4 * math.pi))
    R_V = (3 * Vvol / (4 * math.pi)) ** (1 / 3)
    return A, Vvol, R_A, R_V, triA

def make_icosphere(subdiv=4, R=1.0):
    t = (1 + 5**0.5) / 2
    v = np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],
                  [0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]], float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    f = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],
         [10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],
         [2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    for _ in range(subdiv):
        vl = list(v); mid = {}; nf = []
        def mp(i, j):
            k = tuple(sorted((i, j)))
            if k in mid: return mid[k]
            m = v[i] + v[j]; m /= np.linalg.norm(m); mid[k] = len(vl); vl.append(m); return mid[k]
        for a, b, c in f:
            x = mp(a, b); y = mp(b, c); z = mp(c, a)
            nf += [[a, x, z], [b, y, x], [c, z, y], [x, y, z]]
        v = np.array(vl); f = nf
    return v * R, np.array(f)


# ---------------------------------------------------------------- core solver
def _lib_mean_g(V, F, rho_input, n_sample, seed=0):
    nf = outward_normals(V, F)
    p0, p1, p2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    fc = (p0 + p1 + p2) / 3
    triA = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    idx = np.arange(len(F))
    if n_sample and n_sample < len(F):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(F), n_sample, replace=False, p=triA / triA.sum())
    eps = 1e-4 * np.linalg.norm(V.max(0) - V.min(0))
    pts = (fc[idx] - eps * nf[idx]).tolist()
    poly = pg.Polyhedron(
        polyhedral_source=(V.tolist(), F.tolist()),
        density=rho_input,
        normal_orientation=pg.NormalOrientation.OUTWARDS,
        integrity_check=pg.PolyhedronIntegrity.DISABLE)
    res = pg.evaluate(poly, pts, parallel=True)
    g = np.array([np.linalg.norm(r[1]) for r in res])
    return np.sum(g * triA[idx]) / np.sum(triA[idx])

def _calibrate():
    V, F = make_icosphere(4, 1.0)
    A, Vvol, R_A, R_V, _ = geometry(V, F)
    rho = 1.0 / (G_SI * Vvol)          # makes GM=1
    g = _lib_mean_g(V, F, rho, n_sample=None)
    U = (1.0 / R_A**2) / g
    return U, abs(g * R_A**2 - 1) * 100  # factor, sphere residual %

def mean_surface_gravity(V, F, GM, U):
    A, Vvol, R_A, R_V, _ = geometry(V, F)
    rho = GM / (G_SI * Vvol)
    n_sample = None if len(F) <= 6000 else 3000
    g = _lib_mean_g(V, F, rho, n_sample) * U
    return g


# ---------------------------------------------------------------- driver
def main():
    print("=" * 64)
    print("Shape-based surface gravity solver — full Werner-Scheeres (polyhedral-gravity)")
    print("=" * 64)
    U, sphere_res = _calibrate()
    print(f"Self-test on sphere: residual = {sphere_res:.2f}%  "
          f"(must be small; unit factor U={U:.5f})")
    if sphere_res > 3:
        print("  WARNING: sphere residual large — investigate before trusting results.")
    print()

    if len(sys.argv) < 3:
        print("No shape model given. To run a body:")
        print('    python gravity_ws.py  "model.obj"  GM_km3_s2')
        return

    path, GM = sys.argv[1], float(sys.argv[2])
    V, F = load_obj(path)
    A, Vvol, R_A, R_V, _ = geometry(V, F)
    rho = GM / G_SI / Vvol * 1e-9      # only for the density print-out (g/cm^3-ish)
    print(f"BODY: {path}")
    print(f"  faces={len(F)}  vertices={len(V)}")
    print(f"  A={A:.3f} km^2   V={Vvol:.3f} km^3")
    print(f"  R_A={R_A:.5f} km   R_V={R_V:.5f} km   Psi={A/(4*math.pi*R_V**2):.5f}")

    g = mean_surface_gravity(V, F, GM, U)
    gA, gV = GM / R_A**2, GM / R_V**2
    print(f"\n  <g>_true (Werner-Scheeres) = {g:.6e} km/s^2")
    print(f"  GM/R_A^2 = {gA:.6e}   err = {100*(gA-g)/g:+.2f}%")
    print(f"  GM/R_V^2 = {gV:.6e}   err = {100*(gV-g)/g:+.2f}%")
    print(f"  --> R_A more accurate than R_V: {abs(gA-g) < abs(gV-g)}")
    print("=" * 64)

if __name__ == "__main__":
    main()
