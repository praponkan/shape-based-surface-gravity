#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic contact binaries with tunable geometry, for testing the claim that
the excess mass is the split-invariant quantity.

Why this and not another real body
----------------------------------
The invariance claim held on Itokawa and failed on 67P and Arrokoth. The
proposed explanation is that the invariance is the signature of a real mass
anomaly rather than a property of the method: where the dispersion genuinely
prefers a contrast, the region shrinks and its density rises in compensation,
holding the product fixed; where it does not, the density sits near the bulk
value and the excess mass simply tracks the region volume.

A further real body cannot settle this. We cannot choose in advance whether it
will carry a signal, and we do not know any body's true interior, so agreement
would only mean "it happened twice". A synthetic body lets the signal strength
be dialled from zero to strong while everything else is controlled, so the
threshold can be located rather than guessed.

What is NOT being tested
------------------------
Potential-dispersion minimization does not fit observed data. It assumes the
true density is the one that makes the surface geopotential most uniform.
Imposing an arbitrary density and asking the method to recover it would
therefore fail by construction, and would test nothing. What is tested here is
the structure of the objective function itself.

Body construction
-----------------
Two prolate lobes of revolution about x, unioned:

    r(x) = max_i  A_i * sqrt(1 - ((x - c_i)/a_i)^2)

The lobe amplitudes, half-lengths and separation set the depth of the neck and
the size ratio, and hence how strongly the dispersion prefers a contrast. A
y-scale factor breaks the axial symmetry.
"""

import numpy as np


def lobe_profile(x, lobes):
    r = np.zeros_like(x)
    for A, c, a in lobes:
        t = (x - c) / a
        inside = np.abs(t) < 1.0
        r = np.maximum(r, np.where(inside, A * np.sqrt(np.clip(1 - t * t, 0, 1)), 0.0))
    return r


def build_body(lobes, nx=100, nth=48, yscale=1.0):
    """Watertight triangle mesh of the union of prolate lobes about x."""
    xmin = min(c - a for _, c, a in lobes)
    xmax = max(c + a for _, c, a in lobes)
    x = np.linspace(xmin, xmax, nx)
    r = lobe_profile(x, lobes)

    # Trim the near-zero-radius ends so that only the two apex vertices sit on
    # the axis. A station of radius a few nanometres produces a ring of
    # vertices that are distinct by index but coincident in position to any
    # sensible tolerance, which leaves the mesh non-manifold in effect and
    # generates slivers at the tips. The threshold is relative to the body.
    keep = r > 1e-6 * float(np.max(r))
    first, last = np.argmax(keep), len(keep) - 1 - np.argmax(keep[::-1])
    x, r = x[first:last + 1], r[first:last + 1]

    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    ct, st = np.cos(th), np.sin(th)

    V = [[xmin, 0.0, 0.0]]                       # apex at -x
    for xi, ri in zip(x, r):
        for c_, s_ in zip(ct, st):
            V.append([xi, ri * c_ * yscale, ri * s_])
    V.append([xmax, 0.0, 0.0])                   # apex at +x
    V = np.array(V, float)

    F = []
    n_ring = nth
    for j in range(n_ring):                      # cap at -x
        a = 1 + j
        b = 1 + (j + 1) % n_ring
        F.append([0, b, a])
    for i in range(len(x) - 1):                  # body
        base0 = 1 + i * n_ring
        base1 = 1 + (i + 1) * n_ring
        for j in range(n_ring):
            j2 = (j + 1) % n_ring
            a, b = base0 + j, base0 + j2
            c, d = base1 + j, base1 + j2
            F.append([a, b, d])
            F.append([a, d, c])
    top = len(V) - 1
    base = 1 + (len(x) - 1) * n_ring
    for j in range(n_ring):                      # cap at +x
        a = base + j
        b = base + (j + 1) % n_ring
        F.append([top, a, b])
    F = np.array(F, int)

    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    if np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() < 0:
        F = F[:, ::-1]
    return V, F


def closed_volume(V, F):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0


def is_watertight(F):
    """Every edge must be shared by exactly two faces, in opposite directions."""
    from collections import Counter
    c = Counter()
    for f in F:
        for k in range(3):
            c[(int(f[k]), int(f[(k + 1) % 3]))] += 1
    bad_dup = sum(1 for v in c.values() if v != 1)
    unmatched = sum(1 for e in c if (e[1], e[0]) not in c)
    return bad_dup == 0 and unmatched == 0, bad_dup, unmatched


# --------------------------------------------------------------- geometries
def CONFIGS():
    """Bodies spanning a range of lobe contrast, from near-spherical to
    strongly bilobed. Lengths in km, on the scale of a small asteroid."""
    return [
        ("sphere_like",
         [(0.14, -0.02, 0.20), (0.13, 0.02, 0.19)]),
        ("gentle_waist",
         [(0.14, -0.09, 0.20), (0.13, 0.09, 0.19)]),
        ("clear_neck_equal",
         [(0.14, -0.13, 0.19), (0.14, 0.13, 0.19)]),
        ("clear_neck_unequal",
         [(0.15, -0.13, 0.21), (0.10, 0.15, 0.15)]),
        ("deep_neck_unequal",
         [(0.15, -0.15, 0.20), (0.09, 0.19, 0.13)]),
        ("very_deep_unequal",
         [(0.15, -0.17, 0.20), (0.08, 0.22, 0.12)]),
    ]


if __name__ == "__main__":
    print("SYNTHETIC BODY CONSTRUCTION CHECK")
    print(f"{'name':<20} {'faces':>7} {'verts':>7} {'V km3':>10} "
          f"{'len km':>8} {'neck':>7} {'humps':>15} {'watertight':>11}")
    for name, lobes in CONFIGS():
        V, F = build_body(lobes)
        vol = closed_volume(V, F)
        wt, dup, unm = is_watertight(F)
        xs = np.linspace(min(c - a for _, c, a in lobes),
                         max(c + a for _, c, a in lobes), 400)
        rr = lobe_profile(xs, lobes)
        interior = (xs > xs.min() + 0.2 * np.ptp(xs)) & (xs < xs.max() - 0.2 * np.ptp(xs))
        neck = rr[interior].min()
        h = sorted([A for A, _, _ in lobes], reverse=True)
        print(f"{name:<20} {len(F):>7} {len(V):>7} {vol:>10.6f} "
              f"{np.ptp(xs):>8.3f} {neck:>7.4f} {h[0]:.3f}/{h[1]:.3f}      "
              f"{'yes' if wt else f'NO {dup}/{unm}':>11}")

    print("\nanalytic control: a single prolate spheroid A=0.14, a=0.20")
    V, F = build_body([(0.14, 0.0, 0.20)])
    exact = 4 / 3 * np.pi * 0.20 * 0.14 ** 2
    got = closed_volume(V, F)
    print(f"  meshed {got:.8f}   analytic {exact:.8f}   "
          f"{100*(got/exact-1):+.3f}%  (discretisation)")
