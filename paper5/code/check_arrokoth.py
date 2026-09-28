#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the two-shell Arrokoth model double-count, or hide facets inside itself?

    python check_arrokoth.py arrokoth_porter_2024_v01.obj

The model is supplied as two unjoined shells: Euler characteristic 4 over two
connected components, with no repeated or unmatched directed edges. That is
recorded in the paper, but two consequences were not tested, and a reviewer is
right to ask for them.

    Double counting. A polyhedral volume is a sum of signed tetrahedra over all
    facets. If the two shells overlap in space, the material in the overlap is
    counted twice, so the reported total volume is too large and every density
    derived from it is too small.

    Buried facets. If the shells interpenetrate, some facets lie inside the
    other shell and are not part of the true surface. They still enter the
    area-weighted sum for the dispersion, contaminating the objective with
    potentials evaluated at points that are not on the body.

Both are settled by asking, for each shell, how much of it lies inside the
other. That is what this script measures, by Monte Carlo over the bounding box
and by testing facet centroids for containment. Neither test needs the gravity
library.

Output is a short report and, if anything is found, the size of the effect.
"""

import sys

import numpy as np

from surface_slope_lib import load_obj


def components(F):
    """Split the faces into connected components by shared vertices."""
    adj = {}
    for f in F:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, c), (c, a)):
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    label, comp = {}, 0
    for s in adj:
        if s in label:
            continue
        stack, comp = [s], comp + 1
        while stack:
            u = stack.pop()
            if u in label:
                continue
            label[u] = comp
            stack.extend(w for w in adj[u] if w not in label)
    groups = {}
    for i, f in enumerate(F):
        groups.setdefault(label[int(f[0])], []).append(i)
    return [np.asarray(v) for v in groups.values()]


def signed_volume(V, F):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)


def areas(V, F):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def inside(points, V, F, batch=2000):
    """Ray casting along +x, vectorised over facets.

    A point is inside a closed surface if a ray from it crosses the surface an
    odd number of times. The ray direction is perturbed slightly off-axis so
    that it does not run along an edge.
    """
    d = np.array([1.0, 0.0031, 0.0017])
    d /= np.linalg.norm(d)
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-14
    inv = np.zeros_like(det)
    inv[ok] = 1.0 / det[ok]

    out = np.zeros(len(points), bool)
    for s in range(0, len(points), batch):
        P = points[s:s + batch]
        tv = P[:, None, :] - p0[None, :, :]
        u = np.einsum("ijk,jk->ij", tv, pv) * inv[None, :]
        qv = np.cross(tv, e1[None, :, :])
        v = (qv @ d) * inv[None, :]
        t = np.einsum("ijk,jk->ij", qv, e2) * inv[None, :]
        hit = ok[None, :] & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-12)
        out[s:s + batch] = (hit.sum(axis=1) % 2) == 1
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    n_mc = int(sys.argv[2]) if len(sys.argv) > 2 else 200000

    print(f"Loading {path} ...")
    V, F = load_obj(path)
    print(f"  {len(V)} vertices, {len(F)} facets")

    groups = components(F)
    print(f"\n1. CONNECTED COMPONENTS: {len(groups)}")
    if len(groups) == 1:
        print("   Single component. Nothing further to test; the concerns "
              "below cannot arise.")
        return 0

    vols, Fs = [], []
    for i, idx in enumerate(sorted(groups, key=len, reverse=True), start=1):
        Fi = F[idx]
        Fs.append(Fi)
        v = abs(signed_volume(V, Fi))
        vols.append(v)
        lo, hi = V[np.unique(Fi)].min(0), V[np.unique(Fi)].max(0)
        print(f"   component {i}: {len(Fi):>6} facets, volume {v:>10.2f}, "
              f"x from {lo[0]:>8.3f} to {hi[0]:>8.3f}")
    total = abs(signed_volume(V, F))
    print(f"   sum of components {sum(vols):.2f}, whole-mesh volume "
          f"{total:.2f}, difference {100*(total/sum(vols)-1):+.4f}%")

    if len(groups) != 2:
        print("\n   More than two components; the pairwise tests below cover "
              "only the two largest.")

    A, B = Fs[0], Fs[1]

    # --- 2. intersection volume, by Monte Carlo -------------------------
    print(f"\n2. INTERSECTION VOLUME  ({n_mc} samples)")
    va, vb = np.unique(A), np.unique(B)
    lo = np.maximum(V[va].min(0), V[vb].min(0))
    hi = np.minimum(V[va].max(0), V[vb].max(0))
    if np.any(hi <= lo):
        print("   The bounding boxes do not overlap. The intersection volume "
              "is exactly zero, so no material is double counted.")
        inter = 0.0
    else:
        rng = np.random.default_rng(0)
        P = rng.uniform(lo, hi, size=(n_mc, 3))
        box = float(np.prod(hi - lo))
        both = inside(P, V, A) & inside(P, V, B)
        frac = both.mean()
        inter = frac * box
        se = box * np.sqrt(max(frac * (1 - frac), 1e-12) / n_mc)
        print(f"   overlapping bounding box: {box:.2f} in volume units")
        print(f"   samples inside both shells: {both.sum()} of {n_mc}")
        print(f"   intersection volume {inter:.3f} +/- {se:.3f}, "
              f"{100*inter/total:.4f} +/- {100*se/total:.4f} per cent of the "
              f"total")
        if inter > 3 * se and 100 * inter / total > 0.05:
            print(f"   -> MATERIAL IS DOUBLE COUNTED. The true volume is "
                  f"about {total - inter:.2f}, and every density derived from "
                  f"the reported total is low by {100*inter/total:.2f} per "
                  f"cent.")
        else:
            print("   -> consistent with zero: the shells touch but do not "
                  "interpenetrate, and no material is double counted.")

    # --- 3. buried facets ------------------------------------------------
    print(f"\n3. BURIED FACETS")
    tot_area = areas(V, F).sum()
    buried_n = buried_a = 0
    for lab, Fi, Fj in (("1 inside 2", A, B), ("2 inside 1", B, A)):
        C = (V[Fi[:, 0]] + V[Fi[:, 1]] + V[Fi[:, 2]]) / 3.0
        ins = inside(C, V, Fj)
        a = areas(V, Fi)[ins].sum()
        buried_n += int(ins.sum())
        buried_a += a
        print(f"   component {lab}: {ins.sum():>6} of {len(Fi)} facet "
              f"centroids buried, {100*a/tot_area:.4f} per cent of total area")
    if buried_n == 0:
        print("   -> no facet lies inside the other shell. The area-weighted "
              "sums are taken over the true surface only.")
    else:
        print(f"   -> {buried_n} facets carrying {100*buried_a/tot_area:.3f} "
              f"per cent of the surface area are not on the true surface. "
              f"They enter the dispersion sum and should be removed; rerun "
              f"without them and report the change.")

    # --- 4. what this means for the external validation -------------------
    print(f"\n4. THE PUBLISHED LOBE FRACTION")
    big = max(vols) / sum(vols)
    print(f"   larger component is {100*big:.2f} per cent of the summed "
          f"component volumes")
    print(f"   the paper reports a clipped region of 66.33 per cent against "
          f"66.30 from Porter et al.")
    if abs(100 * big - 66.33) < 0.5:
        print("   -> these are close, so the agreement is largely a property "
              "of the supplied model rather than an independent test of the "
              "clipping. Say so where the validation is claimed.")
    else:
        print("   -> the clipped region differs from a whole component, so "
              "the agreement does test the clipping.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
