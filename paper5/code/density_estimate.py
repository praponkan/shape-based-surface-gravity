#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper V -- Interior density estimation by geopotential-dispersion minimization.

WHAT CHANGED IN THIS VERSION (and why it matters)
-------------------------------------------------
The previous version closed each sub-region with a fan of triangles (O, b, a)
to an interior point O. That construction is watertight and its volume is
exact -- but the solid it encloses is a CONE anchored at O, not the plane-cut
lobe the density model intends. On a unit sphere cut at x = 0.5 the cone has
volume 1.0472 and centroid x = 0.5625, whereas the intended spherical cap has
volume 0.6545 and centroid x = 0.6750: a factor 1.60 in volume and a centroid
displaced by 0.11 R. Raising the "head" density therefore added mass along a
wedge reaching back to the centre of the body instead of adding it in the head,
which reversed the sign of the dispersion trend relative to the Paper IV pilot.

The volume-closure test in that version compared the fan mesh against the sum
of the same tetrahedra, so it could never detect this: it verified internal
consistency, not that the solid was the intended one.

Here regions are built by EXACT PLANE CLIPPING. Each triangle crossing the
plane is split (Sutherland-Hodgman), the intersection points are snapped onto
the plane, and the opening is capped by fanning every boundary edge, traversed
in reverse, to a coplanar apex. Fan triangulation of a planar loop is valid for
any apex in the plane, so non-convex and multiply-connected cross-sections are
handled correctly.

Three independent checks now guard the geometry (run --selftest, and the
predicate check runs automatically on every region):
  1. analytic     -- spherical cap volume and centroid, and a spherical slab
  2. complement   -- V(x > c) + V(x < c) must equal the body volume
  3. predicate    -- every vertex of the region must satisfy the defining
                     inequality. The cone construction fails this outright,
                     since its apex lies at the centre of the body.

Density models
--------------
  uniform : one density, fixed by the total mass. Reference case.
  head    : rho_head for x > x_split, rho_rest elsewhere.
  neck    : rho_neck for |x - x_neck| < w/2, rho_rest elsewhere.

In every case the remaining density follows from
    rho_R * V_R + rho_rest * (V_tot - V_R) = M_total,
so each model has exactly one free parameter.

Usage
-----
    python density_estimate.py <obj> --mtotal <kg> --period <h> [options]

  --model {uniform,head,neck,both}   default: both
  --xsplit <km>       head/body split plane (default: auto, at the neck)
  --neckwidth <km>    neck slab width (default: auto, from the profile)
  --target <N>        decimate to N facets first (default: 20000)
  --scan lo,hi,step   density scan in kg/m^3 (default: 300,4000,25)
  --profile           print the cross-sectional profile and exit
  --profilerecord <f> with --profile, also append the profile and the saddle
                      diagnostics to a JSON file
  --decimator <k>     qem (default, preserves topology) or cluster (the
                      earlier vertex-clustering method, faster but does not)
  --meshcache <dir>   store and reuse decimated meshes in <dir>, so that every
                      run of the same body at the same target uses the
                      identical mesh and the decimation is paid for once
  --scale <f>         multiply every coordinate by f before anything else,
                      so the volume changes by f^3. Use to test how much a
                      shape model's volume offset moves the answer. Note that
                      this is NOT equivalent to rescaling the recovered
                      densities: gravity and the centrifugal term scale
                      differently, so the inversion must be redone.
  --fracsplit <f>     choose the split plane by bisection so that the region
                      holds fraction f of the body volume, instead of giving
                      its position with --xsplit. Makes sweeps comparable
                      across bodies of different shape.
  --cliptol <f>       plane-clipping tolerance as a fraction of the
                      bounding-box diagonal (default 1e-10). Raise to about
                      1e-6 to snap near-plane vertices and remove sliver
                      triangles.
  --record <file>     append a full record of the run to a JSON file
  --selftest          run the geometry validation and exit

Lengths in the shape file are assumed to be km.
"""

import sys
import math
import json
import os
import datetime
import time

import numpy as np

from surface_slope_lib import load_obj, outward_normals, centroids, decimate_cluster


def curvature_sigma(arr, k, frac):
    """Density uncertainty implied by a given fractional rise in dispersion.

    A parabola is fitted to the five scan points centred on the minimum. The
    returned value is how far rho must move to raise the dispersion by `frac`
    of its value at the minimum. It quantifies how sharply the minimum is
    defined; it says NOTHING about whether the split plane was the right one.
    """
    a = max(0, k - 2)
    b = min(len(arr), k + 3)
    if b - a < 3:
        return None
    x, y = arr[a:b, 0], arr[a:b, 2]
    try:
        c = np.polyfit(x, y, 2)
    except Exception:
        return None
    if c[0] <= 0:
        return None
    y0 = np.polyval(c, -c[1] / (2 * c[0]))
    return float(math.sqrt(frac * y0 / c[0]))


def append_record(path, entry):
    """Append one run record to a JSON list, creating the file if needed."""
    data = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                data = [data]
        except Exception:
            backup = path + ".corrupt"
            os.replace(path, backup)
            print(f"  WARNING: {path} was unreadable and was moved to {backup}")
            data = []
    data.append(entry)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return len(data)


# --------------------------------------------------------------------------
# geometry: exact plane clipping
# --------------------------------------------------------------------------
def face_areas(V, F):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def clip_halfspace(V, F, axis, value, keep_greater=True, tol=None, report=False,
                   tol_rel=None):
    """Closed mesh of the part of the solid with x_axis > value (or < value).

    Vertices lying on the cutting plane within `tol` are classified ON, snapped
    exactly onto the plane, and are never duplicated as intersection points.
    Only strict sign changes generate an intersection. Without this, a vertex
    sitting exactly on the plane makes both of its edges register as crossings,
    producing two coincident intersection points and hence zero-area triangles
    on both the surface and the cap. A single zero-area face has a null normal
    and makes the polyhedral gravity evaluation return NaN at EVERY field
    point, not merely near the plane.

    `tol_rel` sets the tolerance as a fraction of the bounding-box diagonal.
    The default of 1e-10 removes exact degeneracies but leaves slivers: a
    vertex a few millimetres from the plane on a kilometre-scale body still
    generates a triangle millimetres across, which the polyhedral gravity
    evaluation flags as numerically unstable. Raising it to about 1e-6 snaps
    those vertices and removes the slivers, at the cost of moving the surface
    by that same tiny distance.
    """
    V = np.asarray(V, float)
    if tol is None:
        tol = (tol_rel if tol_rel is not None else 1e-10) * \
            float(np.linalg.norm(V.max(0) - V.min(0)))

    s = 1.0 if keep_greater else -1.0
    d = s * (V[:, axis] - value)

    code = np.zeros(len(V), int)
    code[d > tol] = 1
    code[d < -tol] = -1

    Vs = V.copy()
    Vs[code == 0, axis] = value          # snap ON vertices exactly

    outV = [v for v in Vs]
    tris = []

    # One intersection point per EDGE, not per facet. The two facets sharing a
    # cut edge must reference the same vertex index, or the surface is torn
    # along the cut: the mesh then has the right volume, since that is a sum of
    # signed tetrahedra, but it is not a closed 2-manifold, and the cap fan is
    # stitched to nothing.
    edge_pt = {}

    def cut_point(ia, ib):
        key = (ia, ib) if ia < ib else (ib, ia)
        j = edge_pt.get(key)
        if j is not None:
            return j
        da, db = d[ia], d[ib]
        t = da / (da - db)
        p = Vs[ia] + t * (Vs[ib] - Vs[ia])
        p[axis] = value
        outV.append(p)
        j = len(outV) - 1
        edge_pt[key] = j
        return j

    for f in F:
        idx = (int(f[0]), int(f[1]), int(f[2]))
        cs = (code[idx[0]], code[idx[1]], code[idx[2]])
        if max(cs) <= 0:
            continue                      # nothing strictly inside
        if min(cs) >= 0:
            tris.append(list(idx))        # nothing strictly outside
            continue

        poly = []
        for k in range(3):
            ia, ib = idx[k], idx[(k + 1) % 3]
            ca, cb = code[ia], code[ib]
            if ca >= 0:
                poly.append(ia)
            if ca * cb < 0:               # strict crossing only
                poly.append(cut_point(ia, ib))

        clean = []
        for j in poly:
            if not clean or np.linalg.norm(outV[j] - outV[clean[-1]]) > tol:
                clean.append(j)
        while len(clean) > 1 and \
                np.linalg.norm(outV[clean[0]] - outV[clean[-1]]) <= tol:
            clean.pop()
        if len(clean) < 3:
            continue
        for k in range(1, len(clean) - 1):
            tris.append([clean[0], clean[k], clean[k + 1]])

    Vc = np.array(outV)

    on_plane = np.abs(Vc[:, axis] - value) <= tol
    cap_edges = []
    for t in tris:
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            if on_plane[a] and on_plane[b] and \
                    np.linalg.norm(Vc[a] - Vc[b]) > tol:
                cap_edges.append((a, b))

    if not cap_edges:
        raise ValueError(f"clipping plane at {value} does not cut the body")

    # Close the opening. The cut boundary may consist of several disjoint
    # loops: a cross-section can be multiply connected, or the region can fall
    # into more than one piece, and on a body supplied as two unjoined shells
    # both happen. Fanning every boundary edge to one apex would then stitch
    # unrelated loops together and produce a surface that is not a manifold,
    # even though its volume remains exact. Each loop is therefore traced and
    # capped separately, with its own apex.
    succ = {}
    for a, b in cap_edges:
        succ.setdefault(a, []).append(b)

    remaining = list(cap_edges)
    used = set()
    loops = []
    for start_edge in cap_edges:
        if start_edge in used:
            continue
        loop = []
        a, b = start_edge
        cur = start_edge
        while cur is not None and cur not in used:
            used.add(cur)
            loop.append(cur)
            nxt = None
            for c in succ.get(cur[1], []):
                if (cur[1], c) not in used:
                    nxt = (cur[1], c)
                    break
            cur = nxt
        if loop:
            loops.append(loop)

    Vc = np.array(outV)
    for loop in loops:
        verts = sorted({i for e in loop for i in e})
        apex = Vc[verts].mean(0)
        apex[axis] = value
        ia = len(Vc)
        Vc = np.vstack([Vc, apex])
        for a, b in loop:          # reverse traversal closes the solid
            tris.append([ia, b, a])

    Fc = np.array(tris, int)

    keep = np.array([len(set(t.tolist())) == 3 for t in Fc])
    A = face_areas(Vc, Fc)
    pos = A[A > 0]
    keep &= A > (1e-12 * np.median(pos) if len(pos) else 0.0)
    n_drop = int((~keep).sum())
    Fc = Fc[keep]

    used, inv = np.unique(Fc, return_inverse=True)
    Vc, Fc = Vc[used], inv.reshape(Fc.shape).astype(int)
    if report:
        return Vc, Fc, n_drop
    return Vc, Fc


def clip_slab(V, F, axis, lo, hi, tol_rel=None):
    Vc, Fc = clip_halfspace(V, F, axis, lo, True, tol_rel=tol_rel)
    return clip_halfspace(Vc, Fc, axis, hi, False, tol_rel=tol_rel)


def closed_volume(V, F):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    return np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0


def closed_centroid(V, F):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    tv = np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0
    tc = (p0 + p1 + p2) / 4.0
    return (tc * tv[:, None]).sum(0) / tv.sum()


def plane_for_fraction(V, F, V_tot, target, iters=34, tol_rel=None):
    """Bisect for the plane leaving `target` of the volume above it.

    Selecting planes by volume fraction rather than by position makes sweeps
    comparable across bodies of very different shape. A plane moved a fixed
    distance sweeps very little volume on a body with a long thin neck and a
    great deal on a compact one, so equal distances do not probe the same
    thing.
    """
    xlo, xhi = float(V[:, 0].min()), float(V[:, 0].max())
    for _ in range(iters):
        xm = 0.5 * (xlo + xhi)
        try:
            Vr, Fr = clip_halfspace(V, F, 0, xm, True, tol_rel=tol_rel)
            f = abs(closed_volume(Vr, Fr)) / V_tot
        except ValueError:
            f = 0.0
        if f > target:
            xlo = xm
        else:
            xhi = xm
    return 0.5 * (xlo + xhi)


def _load_decimator(kind):
    """Return (function, label). Falls back to clustering if QEM is absent."""
    if kind == "cluster":
        return decimate_cluster, "cluster"
    try:
        from decimate_qem import decimate_qem
        return decimate_qem, "qem"
    except ImportError:
        print("  NOTE: decimate_qem.py not found; falling back to vertex "
              "clustering, which does not preserve topology")
        return decimate_cluster, "cluster"


def clean_mesh(V, F, area_rel=1e-12):
    """Remove the debris that vertex-clustering decimation leaves behind.

    Snapping vertices to a grid and rebuilding faces from the snapped indices
    collapses some faces onto two vertices and can map two originally distinct
    faces onto the same vertex triple. Both leave repeated directed edges. The
    pass below removes degenerate faces, duplicate faces and slivers, which is
    cheap and cannot make the topology worse; whether it restores a closed
    surface depends on whether the decimator also created non-manifold vertex
    configurations, which no face-level cleaning can repair.

    Returns (V, F, report).
    """
    F = np.asarray(F, int)
    n0 = len(F)

    keep = np.array([len(set(t.tolist())) == 3 for t in F])
    n_degen = int((~keep).sum())
    F = F[keep]

    seen, keep = set(), []
    for t in F:
        k = tuple(sorted(int(v) for v in t))
        keep.append(k not in seen)
        seen.add(k)
    keep = np.array(keep)
    n_dup = int((~keep).sum())
    F = F[keep]

    A = face_areas(V, F)
    pos = A[A > 0]
    thr = area_rel * np.median(pos) if len(pos) else 0.0
    keep = A > thr
    n_sliver = int((~keep).sum())
    F = F[keep]

    used, inv = np.unique(F, return_inverse=True)
    return (V[used], inv.reshape(F.shape).astype(int),
            {"faces_before": n0, "faces_after": len(F),
             "degenerate_removed": n_degen, "duplicate_removed": n_dup,
             "sliver_removed": n_sliver})


def check_manifold(F):
    """Is the mesh a closed, consistently oriented 2-manifold?

    Every directed edge must appear exactly once, and its reverse must be
    present. A mesh can have an exact volume and still fail this: the
    divergence-theorem volume is a sum of signed tetrahedra and needs only
    that the oriented facets tile the boundary with the right signs, not that
    they are stitched together. A torn surface therefore passes every volume
    and predicate test while being, topologically, not a surface at all.

    Returns (ok, diagnostics). The Euler characteristic is 2 for a single
    topological sphere and 2k for k disjoint ones, so it is judged against the
    component count rather than against 2. Some shape models arrive as several
    unjoined shells; Arrokoth is supplied that way, and a region cut from such
    a body is legitimately in more than one piece.
    """
    from collections import Counter
    c = Counter()
    for f in F:
        for k in range(3):
            c[(int(f[k]), int(f[(k + 1) % 3]))] += 1
    repeated = sum(1 for v in c.values() if v != 1)
    unmatched = sum(1 for (a, b) in c if (b, a) not in c)
    n_v = len(np.unique(np.asarray(F)))
    n_e = len(c) // 2
    n_f = len(F)
    euler = n_v - n_e + n_f
    # Connected components, so that chi can be judged against the right
    # target: a mesh in k pieces, each a topological sphere, has chi = 2k. A
    # shape model supplied as two unjoined shells is the case in point.
    adj = {}
    for f in F:
        a, b, cc = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, cc), (cc, a)):
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    seen, comps = set(), 0
    for s0 in adj:
        if s0 in seen:
            continue
        comps += 1
        stack = [s0]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(v for v in adj[u] if v not in seen)

    closed = (repeated == 0 and unmatched == 0)
    ok = closed and euler == 2 * comps
    return ok, {"repeated_directed_edges": repeated,
                "unmatched_directed_edges": unmatched,
                "euler_characteristic": int(euler),
                "components": int(comps),
                "expected_euler": int(2 * comps),
                "closed": bool(closed),
                "vertices": int(n_v), "edges": int(n_e), "faces": int(n_f)}


def check_predicate(Vr, axis, lo=None, hi=None, tol=1e-9):
    """Largest violation of the inequality that defines the region."""
    x = Vr[:, axis]
    worst = 0.0
    if lo is not None:
        worst = max(worst, float(np.max(lo - x)))
    if hi is not None:
        worst = max(worst, float(np.max(x - hi)))
    return max(worst, 0.0), worst <= tol


# --------------------------------------------------------------------------
# neck detection
# --------------------------------------------------------------------------
def cross_section_profile(V, F, O, nbins=60):
    """Outer half-width perpendicular to x, binned along x."""
    C = centroids(V, F)
    x = C[:, 0]
    r = np.hypot(C[:, 1] - O[1], C[:, 2] - O[2])
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(nbins, np.nan)
    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        if m.sum() >= 3:
            prof[i] = np.percentile(r[m], 90)
    return mid, prof


def find_neck(mid, prof, edge_frac=0.20, min_prominence=0.02):
    """Locate a genuine saddle in the cross-section profile.

    A neck is a LOCAL minimum with a hump on either side, not merely the
    smallest half-width in the interior of the profile. The distinction
    matters: a body that tapers at both ends has its smallest interior
    half-width on a shoulder, and taking the absolute minimum then returns a
    plausible number that is not a saddle at all. On Eros the absolute-minimum
    rule selects x = -10.76 km (half-width 5.25 km) on the flank, while the
    true saddle at Himeros is at x = +4.47 km (half-width 6.08 km) between
    humps of 8.35 and 7.94 km.

    Returns (x_neck, slab_width, half_width, diagnostics). If no qualifying
    saddle exists, x_neck is None and the caller must supply a plane; the
    diagnostics record every candidate found and why it was rejected.
    """
    good = np.isfinite(prof)
    m, p = mid[good], prof[good]
    lo, hi = m.min(), m.max()
    span = hi - lo
    scale = float(np.nanmax(p))

    inner = (m > lo + edge_frac * span) & (m < hi - edge_frac * span)
    idx = np.where(inner)[0]
    if len(idx) < 3:
        idx = np.arange(len(m))

    # Local minima, detected on runs of equal value rather than single points.
    # A symmetric profile puts two equal samples either side of the true
    # minimum, so a strict point test finds nothing at all; plateaux fail the
    # same way.
    lo_i, hi_i = int(idx[0]), int(idx[-1])
    tol_eq = 1e-12 * scale
    cands = []
    j = 0
    while j < len(m):
        k = j
        while k + 1 < len(m) and abs(p[k + 1] - p[j]) <= tol_eq:
            k += 1
        if j > 0 and k < len(m) - 1 and p[j - 1] > p[j] and p[k + 1] > p[k]:
            centre = 0.5 * (m[j] + m[k])
            if lo_i <= (j + k) // 2 <= hi_i:
                left_max = float(p[:j].max())
                right_max = float(p[k + 1:].max())
                prom = (min(left_max, right_max) - p[j]) / scale
                cands.append({"x": float(centre), "half_width": float(p[j]),
                              "prominence": float(prom),
                              "left_hump": left_max,
                              "right_hump": right_max})
        j = k + 1

    diag = {"n_local_minima": len(cands), "candidates": cands,
            "min_prominence": min_prominence,
            "absolute_minimum_x": float(m[idx][int(np.argmin(p[idx]))]),
            "absolute_minimum_half_width": float(p[idx].min())}

    kept = [c for c in cands if c["prominence"] >= min_prominence]
    diag["n_accepted"] = len(kept)
    if not kept:
        return None, None, None, diag

    best = max(kept, key=lambda c: c["prominence"])
    diag["selected"] = best
    x_neck, r_neck = best["x"], best["half_width"]

    thr = 1.15 * r_neck
    left = m[(m < x_neck) & (p > thr)]
    right = m[(m > x_neck) & (p > thr)]
    xl = left.max() if len(left) else lo
    xr = right.min() if len(right) else hi
    return x_neck, max(xr - xl, 0.05 * span), r_neck, diag


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------
def unit_potential(V, F, pts_m):
    """Surface potential of the closed mesh (V, F) at unit density, SI."""
    import polyhedral_gravity as pg
    poly = pg.Polyhedron(
        polyhedral_source=((V * 1000.0).tolist(), F.tolist()),
        density=1.0,
        normal_orientation=pg.NormalOrientation.OUTWARDS,
        integrity_check=pg.PolyhedronIntegrity.DISABLE)
    res = pg.evaluate(poly, pts_m.tolist(), parallel=True)
    return -np.array([r[0] for r in res])   # library returns +|U|; physical U < 0


def dispersion(U_grav, pot_rot, area):
    U = U_grav + pot_rot
    Ubar = np.sum(U * area) / np.sum(area)
    var = np.sum(((U - Ubar) ** 2) * area) / np.sum(area)
    return math.sqrt(var) / abs(Ubar)


# --------------------------------------------------------------------------
# geometry self-test against closed-form solids
# --------------------------------------------------------------------------
def icosphere(subdiv, R=1.0):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], int)
    for _ in range(subdiv):
        mid, newF, Vl = {}, [], V.tolist()

        def mp(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                mid[k] = len(Vl)
                Vl.append(((V[a] + V[b]) / 2).tolist())
            return mid[k]

        for f in F:
            a, b, c = f
            ab, bc, ca = mp(a, b), mp(b, c), mp(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V = np.array(Vl)
        F = np.array(newF, int)
    return V / np.linalg.norm(V, axis=1)[:, None] * R, F


def selftest():
    R, XS = 1.0, 0.5
    V, F = icosphere(5, R)
    h = R - XS
    vtot = closed_volume(V, F)

    Vc, Fc = clip_halfspace(V, F, 0, XS, True)
    vol, cen = closed_volume(Vc, Fc), closed_centroid(Vc, Fc)
    vol_x = math.pi * h ** 2 * (3 * R - h) / 3
    cen_x = 3 * (2 * R - h) ** 2 / (4 * (3 * R - h))

    Vb, Fb = clip_halfspace(V, F, 0, XS, False)
    volb = closed_volume(Vb, Fb)

    lo, hi = -0.2, 0.2
    Vs, Fs = clip_slab(V, F, 0, lo, hi)
    vs = closed_volume(Vs, Fs)
    vs_x = math.pi * (R ** 2 * (hi - lo) - (hi ** 3 - lo ** 3) / 3)

    _, ok1 = check_predicate(Vc, 0, lo=XS)
    _, ok2 = check_predicate(Vs, 0, lo=lo, hi=hi)
    mf1, d1 = check_manifold(Fc)
    mf2, d2 = check_manifold(Fs)
    comp = abs(vol + volb - vtot) / abs(vtot)

    print("GEOMETRY SELF-TEST  (unit sphere; mesh discretization ~0.05%)")
    print(f"  cap volume    {vol:.6f}  analytic {vol_x:.6f}  "
          f"{100*(vol-vol_x)/vol_x:+.3f}%")
    print(f"  cap centroid  {cen[0]:.6f}  analytic {cen_x:.6f}  "
          f"{100*(cen[0]-cen_x)/cen_x:+.3f}%")
    print(f"  slab volume   {vs:.6f}  analytic {vs_x:.6f}  "
          f"{100*(vs-vs_x)/vs_x:+.3f}%")
    print(f"  complement sum error   {comp*100:.2e}%")
    print(f"  predicate  half-space  {'PASS' if ok1 else 'FAIL'}")
    print(f"  predicate  slab        {'PASS' if ok2 else 'FAIL'}")
    print(f"  manifold   half-space  chi = {d1['euler_characteristic']}, "
          f"unmatched {d1['unmatched_directed_edges']}  "
          f"{'PASS' if mf1 else 'FAIL'}")
    print(f"  manifold   slab        chi = {d2['euler_characteristic']}, "
          f"unmatched {d2['unmatched_directed_edges']}  "
          f"{'PASS' if mf2 else 'FAIL'}")

    tol = 0.5    # per cent; dominated by the polyhedral approximation
    ok = (ok1 and ok2 and mf1 and mf2 and comp < 1e-12
          and abs(100 * (vol - vol_x) / vol_x) < tol
          and abs(100 * (vs - vs_x) / vs_x) < tol)
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
def arg(name, default=None, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)

    obj = sys.argv[1]
    M_total = arg("--mtotal", None, float)
    period_h = arg("--period", None, float)
    model = arg("--model", "both")
    target = arg("--target", 20000, int)
    lo, hi, step = (float(v) for v in arg("--scan", "300,4000,25").split(","))
    tol_rel = arg("--cliptol", None, float)

    if M_total is None or period_h is None:
        sys.exit("need --mtotal <kg> and --period <hours>")

    print(f"Loading {obj} ...")
    V, F = load_obj(obj)
    n_native = len(F)

    scale = arg("--scale", None, float)
    if scale is not None:
        if scale <= 0:
            sys.exit("--scale must be positive")
        V = V * scale
        print(f"  linear scale factor {scale:.6f} applied "
              f"(volume x {scale ** 3:.6f})")
    print(f"  facets {len(F)}")
    dec_kind = arg("--decimator", "qem")
    cache_dir = arg("--meshcache", None)
    if len(F) > target:
        fn, label = _load_decimator(dec_kind)

        # Cache the decimated mesh. Quadric decimation of a million-facet
        # model takes about ten minutes, and every run of the same body at the
        # same target must use the identical mesh: recomputing it is not only
        # slow but, if the decimator were ever made non-deterministic, would
        # silently change the mesh between runs of the same series.
        cpath = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(obj))[0]
            cpath = os.path.join(cache_dir, f"{stem}__{label}_{target}.npz")

        if cpath and os.path.exists(cpath):
            z = np.load(cpath)
            V, F = z["V"], z["F"]
            print(f"  decimated mesh loaded from cache -> {len(F)} "
                  f"({label}, target {target})")
        else:
            t_dec = time.time()
            V, F = fn(V, F, target)
            print(f"  decimated -> {len(F)} by {label} "
                  f"({time.time() - t_dec:.0f} s)")
            if cpath:
                np.savez_compressed(cpath, V=V, F=F)
                print(f"  cached to {cpath}")
        V, F, cl = clean_mesh(V, F)
        if cl["faces_before"] != cl["faces_after"]:
            print(f"  cleaned -> {cl['faces_after']} "
                  f"({cl['degenerate_removed']} degenerate, "
                  f"{cl['duplicate_removed']} duplicate, "
                  f"{cl['sliver_removed']} sliver faces removed)")
    else:
        dec_kind = "none"

    src_ok, src_mf = check_manifold(F)
    print(f"  mesh as used: chi = {src_mf['euler_characteristic']}, "
          f"{src_mf['components']} component(s), "
          f"{src_mf['repeated_directed_edges']} repeated and "
          f"{src_mf['unmatched_directed_edges']} unmatched directed edges  "
          f"{'closed' if src_mf['closed'] else 'NOT CLOSED'}")
    if not src_mf["closed"]:
        print("  WARNING: the source mesh is not a closed surface. Volumes "
              "and potentials remain well defined, since both are sums over "
              "oriented facets, but the region topology inherits the defect.")

    V_tot = abs(closed_volume(V, F))
    C = centroids(V, F)
    O = V.mean(0)
    rho_bulk = M_total / (V_tot * 1e9)
    print(f"  volume {V_tot:.5e} km^3   bulk density {rho_bulk:.1f} kg/m^3")
    print(f"  body centroid x = {closed_centroid(V, F)[0]:+.4f} km")

    mid, prof = cross_section_profile(V, F, O)
    x_neck, w_auto, r_neck, ndiag = find_neck(mid, prof)

    if x_neck is None:
        print(f"  NO NECK FOUND: {ndiag['n_local_minima']} local minima in the "
              f"cross-section profile, none with prominence >= "
              f"{ndiag['min_prominence']:.3f}")
        print(f"  (the smallest interior half-width is "
              f"{ndiag['absolute_minimum_half_width']:.4f} km at x = "
              f"{ndiag['absolute_minimum_x']:.4f} km, but it is not a saddle)")
        print("  the split plane must be supplied with --xsplit")
    else:
        sel = ndiag["selected"]
        print(f"  neck at x = {x_neck:.4f} km (half-width {r_neck:.4f} km), "
              f"slab width {w_auto:.4f} km")
        print(f"  saddle prominence {sel['prominence']:.3f} between humps of "
              f"{sel['left_hump']:.4f} and {sel['right_hump']:.4f} km "
              f"({ndiag['n_accepted']} of {ndiag['n_local_minima']} local "
              f"minima accepted)")
        if abs(ndiag["absolute_minimum_x"] - x_neck) > 1e-9:
            print(f"  NOTE: the absolute interior minimum is elsewhere, at "
                  f"x = {ndiag['absolute_minimum_x']:.4f} km "
                  f"({ndiag['absolute_minimum_half_width']:.4f} km); it is a "
                  f"flank, not a saddle, and has been rejected")

    if "--profile" in sys.argv:
        pmax = np.nanmax(prof)
        print(f"\n  {'x (km)':>10} {'half-width':>12}")
        for m_, p_ in zip(mid, prof):
            if np.isfinite(p_):
                print(f"  {m_:>10.4f} {p_:>12.4f}  {'#' * int(60 * p_ / pmax)}")

        prof_path = arg("--profilerecord", None)
        if prof_path:
            good = np.isfinite(prof)
            entry = {
                "timestamp": datetime.datetime.now().isoformat(
                    timespec="seconds"),
                "command": " ".join(sys.argv),
                "shape_model": obj,
                "facets_native": int(n_native),
                "facets_used": int(len(F)),
                "volume_km3": float(V_tot),
                "body_centroid_x_km": float(closed_centroid(V, F)[0]),
                "x_km": [round(float(v), 6) for v in mid[good]],
                "half_width_km": [round(float(v), 6) for v in prof[good]],
                "neck": {
                    "found": x_neck is not None,
                    "x_neck_km": (float(x_neck) if x_neck is not None
                                  else None),
                    "half_width_km": (float(r_neck) if r_neck is not None
                                      else None),
                    "slab_width_km": (float(w_auto) if w_auto is not None
                                      else None),
                    "prominence": (ndiag["selected"]["prominence"]
                                   if x_neck is not None else None),
                    "left_hump_km": (ndiag["selected"]["left_hump"]
                                     if x_neck is not None else None),
                    "right_hump_km": (ndiag["selected"]["right_hump"]
                                      if x_neck is not None else None),
                    "n_local_minima": ndiag["n_local_minima"],
                    "n_accepted": ndiag["n_accepted"],
                    "min_prominence_threshold": ndiag["min_prominence"],
                    "absolute_interior_minimum_x_km":
                        ndiag["absolute_minimum_x"],
                    "absolute_interior_minimum_half_width_km":
                        ndiag["absolute_minimum_half_width"],
                    "flank_rejected": (
                        x_neck is not None
                        and abs(ndiag["absolute_minimum_x"] - x_neck) > 1e-9),
                    "candidates": ndiag["candidates"],
                },
            }
            n = append_record(prof_path, entry)
            print(f"\n  profile recorded to {prof_path} "
                  f"(entry {n} in the file)")
        return

    record_path = arg("--record", None)
    record = None
    if record_path:
        record = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "shape_model": obj,
            "facets_native": int(n_native),
            "facets_used": int(len(F)),
            "decimation_target": target,
            "decimation_applied": bool(n_native > target),
            "M_total_kg": M_total,
            "period_h": period_h,
            "volume_km3": float(V_tot),
            "linear_scale_applied": (scale if scale is not None else 1.0),
            "bulk_density_kg_m3": float(rho_bulk),
            "body_centroid_x_km": float(closed_centroid(V, F)[0]),
            "scan_lo": lo, "scan_hi": hi, "scan_step": step,
            "clip_tol_rel": (tol_rel if tol_rel is not None else 1e-10),
            "neck_detection": {
                "found": x_neck is not None,
                "x_neck_km": (float(x_neck) if x_neck is not None else None),
                "half_width_km": (float(r_neck) if r_neck is not None else None),
                "slab_width_km": (float(w_auto) if w_auto is not None else None),
                "n_local_minima": ndiag["n_local_minima"],
                "n_accepted": ndiag["n_accepted"],
                "min_prominence_threshold": ndiag["min_prominence"],
                "prominence": (ndiag["selected"]["prominence"]
                               if x_neck is not None else None),
                "left_hump_km": (ndiag["selected"]["left_hump"]
                                 if x_neck is not None else None),
                "right_hump_km": (ndiag["selected"]["right_hump"]
                                  if x_neck is not None else None),
                "absolute_interior_minimum_x_km":
                    ndiag["absolute_minimum_x"],
                "absolute_interior_minimum_half_width_km":
                    ndiag["absolute_minimum_half_width"],
                "flank_rejected": (x_neck is not None and
                                   abs(ndiag["absolute_minimum_x"] - x_neck) > 1e-9),
            },
            "models": {},
        }

    x_split = arg("--xsplit", x_neck, float)
    w_neck = arg("--neckwidth", w_auto, float)

    frac_target = arg("--fracsplit", None, float)
    if frac_target is not None:
        if not 0.0 < frac_target < 1.0:
            sys.exit("--fracsplit takes a volume fraction between 0 and 1")
        x_split = plane_for_fraction(V, F, V_tot, frac_target,
                                     tol_rel=tol_rel)
        Vc, Fc = clip_halfspace(V, F, 0, x_split, True, tol_rel=tol_rel)
        f_got = abs(closed_volume(Vc, Fc)) / V_tot
        print(f"  --fracsplit {frac_target:.4f}: plane at x = {x_split:.6f} km "
              f"gives {100 * f_got:.4f}% of the volume")
        if record is not None:
            record["fracsplit_target"] = frac_target
            record["fracsplit_achieved"] = float(f_got)
            record["fracsplit_x_km"] = float(x_split)

    if x_split is None:
        sys.exit("no neck was found and no --xsplit was given: "
                 "the split plane is undetermined")
    if model in ("neck", "both") and x_neck is None:
        print("  neck model skipped: no saddle was detected")
        model = "head"

    nf = outward_normals(V, F)
    Cm = C * 1000.0
    ext = np.linalg.norm((V.max(0) - V.min(0)) * 1000.0)
    pts = Cm - 1e-4 * ext * nf
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    om = 2 * math.pi / (period_h * 3600.0)
    pot_rot = -0.5 * om ** 2 * (Cm[:, 0] ** 2 + Cm[:, 1] ** 2)

    print("\n  precomputing unit-density field of the whole body ...")
    U_full = unit_potential(V, F, pts)
    d_uniform = dispersion(rho_bulk * U_full, pot_rot, area)
    print(f"  uniform-density reference: dispersion = {d_uniform:.6f}")

    models = []
    if model in ("head", "both"):
        models.append(("head", dict(lo=x_split, hi=None),
                       f"x > {x_split:.4f} km"))
    if model in ("neck", "both"):
        models.append(("neck",
                       dict(lo=x_neck - 0.5 * w_neck, hi=x_neck + 0.5 * w_neck),
                       f"|x - {x_neck:.4f}| < {0.5 * w_neck:.4f} km"))

    for name, sel, desc in models:
        r_lo, r_hi = sel["lo"], sel["hi"]
        rec = None
        if record is not None:
            rec = {"definition": desc, "x_lo_km": r_lo, "x_hi_km": r_hi,
                   "status": "ok"}
            record["models"][name] = rec

        if r_hi is None:
            Vr, Fr, n_drop = clip_halfspace(V, F, 0, r_lo, True, report=True,
                                            tol_rel=tol_rel)
        else:
            Vr, Fr = clip_slab(V, F, 0, r_lo, r_hi, tol_rel=tol_rel)
            n_drop = 0

        V_R = abs(closed_volume(Vr, Fr))
        V_rest = V_tot - V_R
        cen_R = closed_centroid(Vr, Fr)
        viol, ok = check_predicate(Vr, 0, lo=r_lo, hi=r_hi)

        print(f"\n=== {name.upper()} model: {desc} ===")
        print(f"  region {len(Fr)} faces, V = {V_R:.5e} km^3 "
              f"({100 * V_R / V_tot:.1f}% of body)")
        print(f"  region centroid x = {cen_R[0]:+.4f} km")
        print(f"  predicate check: max violation {viol:.2e} km  "
              f"{'PASS' if ok else 'FAIL'}")

        if rec is not None:
            rec.update({
                "region_faces": int(len(Fr)),
                "V_R_km3": float(V_R),
                "volume_fraction_pct": float(100 * V_R / V_tot),
                "complement_fraction_pct": float(100 * V_rest / V_tot),
                "amplification_VR_over_Vrest": float(V_R / V_rest),
                "region_centroid_x_km": float(cen_R[0]),
                "predicate_max_violation_km": float(viol),
                "predicate": "PASS" if ok else "FAIL",
                "degenerate_faces_removed": int(n_drop),
            })

        if not ok:
            print("  ABORT: region is not the intended solid")
            if rec is not None:
                rec["status"] = "ABORT: predicate failed"
            continue

        # independent cross-check: complement volumes must sum to the body
        if r_hi is None:
            Vk, Fk = clip_halfspace(V, F, 0, r_lo, False, tol_rel=tol_rel)
            comp = abs(V_R + abs(closed_volume(Vk, Fk)) - V_tot) / V_tot
            print(f"  complement closure: {comp * 100:.2e}%")
            if rec is not None:
                rec["complement_closure_pct"] = float(comp * 100)

        mf_ok, mf = check_manifold(Fr)
        print(f"  manifold check: chi = {mf['euler_characteristic']} "
              f"(expected {mf['expected_euler']} for "
              f"{mf['components']} component(s)), "
              f"repeated {mf['repeated_directed_edges']}, unmatched "
              f"{mf['unmatched_directed_edges']}  "
              f"{'PASS' if mf_ok else 'FAIL'}")
        if rec is not None:
            rec["manifold"] = mf
            rec["manifold_pass"] = bool(mf_ok)
            rec["source_manifold"] = src_mf
        if not mf_ok:
            # A defect inherited from the source is reported and tolerated;
            # one introduced by the clipping is not. The distinction matters
            # because the potential is a sum over oriented facets and does not
            # consult connectivity: a torn surface still yields the correct
            # field, as measured, but it signals that something is wrong and
            # should not pass silently.
            inherited = not src_mf["closed"]
            tag = "inherited from the source mesh" if inherited else \
                  "introduced by the region construction"
            print(f"  WARNING: the region mesh is not closed, {tag}. Volume "
                  f"and potential are unaffected, both being sums over "
                  f"oriented facets, but the result is flagged.")
            if rec is not None:
                rec["manifold_warning"] = tag

        A_R = face_areas(Vr, Fr)
        print(f"  min face area {A_R.min():.3e} km^2, "
              f"degenerate faces removed {n_drop}")
        if rec is not None:
            rec["min_face_area_km2"] = float(A_R.min())
        if A_R.min() <= 0:
            print("  ABORT: region mesh still carries a zero-area face")
            if rec is not None:
                rec["status"] = "ABORT: zero-area face"
            continue

        U_R = unit_potential(Vr, Fr, pts)
        n_bad = int((~np.isfinite(U_R)).sum())
        if n_bad:
            print(f"  ABORT: U_R has {n_bad} non-finite entries of {len(U_R)}. "
                  "The region mesh is singular at one or more field points; "
                  "results for this model would be meaningless.")
            if rec is not None:
                rec["status"] = f"ABORT: {n_bad} non-finite entries in U_R"
            continue

        rows = []
        for rho_R in np.arange(lo, hi + 0.5 * step, step):
            rho_rest = (M_total - rho_R * V_R * 1e9) / (V_rest * 1e9)
            if rho_rest <= 0:
                continue
            U = rho_rest * U_full + (rho_R - rho_rest) * U_R
            rows.append((rho_R, rho_rest, dispersion(U, pot_rot, area)))
        arr = np.array(rows)
        if not np.isfinite(arr[:, 2]).any():
            print("  ABORT: every dispersion value is non-finite")
            if rec is not None:
                rec["status"] = "ABORT: all dispersions non-finite"
            continue
        k = int(np.nanargmin(arr[:, 2]))
        interior = 0 < k < len(arr) - 1

        print(f"  {'rho_R':>9} {'rho_rest':>10} {'dispersion':>13}")
        stride = max(1, len(arr) // 18)
        for i in range(0, len(arr), stride):
            print(f"  {arr[i,0]:>9.0f} {arr[i,1]:>10.0f} {arr[i,2]:>13.6f}")
        print("  " + "-" * 36)
        print(f"  minimum: rho_{name} = {arr[k,0]:.0f}, "
              f"rho_rest = {arr[k,1]:.0f} kg/m^3, dispersion = {arr[k,2]:.6f}")
        print(f"  contrast rho_{name}/rho_rest = {arr[k,0]/arr[k,1]:.3f}")
        print(f"  INTERIOR MINIMUM: {'YES' if interior else 'NO (at scan edge)'}")
        print(f"  improvement over uniform: "
              f"{100 * (d_uniform - arr[k,2]) / d_uniform:+.3f}%")

        if rec is not None:
            rho_R, rho_rest, dmin = float(arr[k, 0]), float(arr[k, 1]), float(arr[k, 2])
            excess = (rho_R - rho_rest) * V_R * 1e9
            com = excess * (cen_R[0] - closed_centroid(V, F)[0]) / M_total * 1000.0
            rec.update({
                "scan": [{"rho_R": round(float(a), 4),
                          "rho_rest": round(float(b), 4),
                          "dispersion": round(float(c), 9)}
                         for a, b, c in rows],
                "n_scan_points": len(rows),
                "minimum": {
                    "rho_R": rho_R, "rho_rest": rho_rest,
                    "dispersion": dmin,
                    "contrast": rho_R / rho_rest,
                    "interior_minimum": bool(interior),
                    "improvement_over_uniform_pct":
                        100 * (d_uniform - dmin) / d_uniform,
                    "index_in_scan": k,
                },
                "derived": {
                    "excess_mass_kg": float(excess),
                    "excess_mass_pct_of_total": float(100 * excess / M_total),
                    "com_offset_m": float(com),
                    "sigma_rho_at_0p60pct": curvature_sigma(arr, k, 0.0060),
                    "sigma_rho_at_5p9pct": curvature_sigma(arr, k, 0.059),
                },
            })
            s = rec["derived"]["sigma_rho_at_0p60pct"]
            if s:
                print(f"  sigma(rho) from the curvature: +/-{s:.0f} kg/m^3 "
                      f"at 0.60% noise ({100*s/rho_R:.1f}% of rho)")
                print(f"  excess mass {excess:.4e} kg "
                      f"({100*excess/M_total:.3f}% of total), "
                      f"COM offset {com:+.2f} m")

    if record is not None:
        record["uniform_reference_dispersion"] = float(d_uniform)
        n = append_record(record_path, record)
        print(f"\n  recorded to {record_path} (run {n} in the file)")


if __name__ == "__main__":
    main()
