#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Do the buried facets of the Arrokoth model change the answer?

    python rerun_arrokoth_clean.py arrokoth_porter_2024_v01.obj

The shape model is supplied as two unjoined shells. They touch but do not
interpenetrate --- the intersection is 0.0445 per cent of the volume, which is
consistent with zero --- so nothing is double counted and the volumes and
densities reported in the paper stand.

But 455 facets, carrying 0.943 per cent of the surface area, have their
centroids inside the other shell. Those facets are not on the true surface.
They still enter the area-weighted sums that define

    sigma_U = sd(U) / |mean(U)|

so the objective is being evaluated partly at points that lie within the body.
The effect ought to be small, at under one per cent of the area, but on a body
whose recovered contrast crosses unity within the plausible density range that
is not an assumption worth making. This script measures it.

Method
------
Identify the buried facets by testing each centroid for containment in the
other component, using the same ray-casting test as check_arrokoth.py. Then run
the inversion twice at each plane, once over all facets and once over the
surviving ones, and report the difference.

Removing the facets changes only which field points enter the average. The
polyhedron itself is untouched, so the volume, the mass and the region geometry
are identical between the two runs, and the comparison isolates one effect.

Output goes to arrokoth_clean.json.
"""

import argparse
import json
import sys
import time

import numpy as np

import density_estimate as de
from surface_slope_lib import load_obj, outward_normals


def components(F):
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


def inside(points, V, F, batch=1500):
    """Ray casting along a direction slightly off every axis."""
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
        hit = (ok[None, :] & (u >= 0) & (u <= 1) & (v >= 0)
               & (u + v <= 1) & (t > 1e-12))
        out[s:s + batch] = (hit.sum(axis=1) % 2) == 1
    return out


def find_buried(V, F):
    """Which facets have their centroid inside another component?"""
    groups = components(F)
    if len(groups) < 2:
        return np.zeros(len(F), bool), groups
    buried = np.zeros(len(F), bool)
    for i, gi in enumerate(groups):
        C = (V[F[gi, 0]] + V[F[gi, 1]] + V[F[gi, 2]]) / 3.0
        for j, gj in enumerate(groups):
            if i == j:
                continue
            buried[gi[inside(C, V, F[gj])]] = True
    return buried, groups


def invert(V, F, mask, M_tot, period_h, x_split, scan):
    """Scan rho_R, averaging only over the facets selected by `mask`."""
    V_tot = abs(de.closed_volume(V, F))
    Vr, Fr = de.clip_halfspace(V, F, 0, x_split, True)
    V_R = abs(de.closed_volume(Vr, Fr))

    nf = outward_normals(V, F)
    C = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    ext = float(np.linalg.norm(V.max(0) - V.min(0)))
    pts_m = (C - 1e-4 * ext * nf) * 1000.0
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)

    U_full = de.unit_potential(V, F, pts_m)
    U_R = de.unit_potential(Vr, Fr, pts_m)
    om = 2.0 * np.pi / (period_h * 3600.0)
    U_rot = -0.5 * om ** 2 * (pts_m[:, 0] ** 2 + pts_m[:, 1] ** 2)

    Am = A[mask]
    w = Am / Am.sum()
    Uf, Ur, Uo = U_full[mask], U_R[mask], U_rot[mask]

    def disp(rho_R):
        rho_rest = (M_tot - rho_R * V_R * 1e9) / ((V_tot - V_R) * 1e9)
        U = rho_rest * Uf + (rho_R - rho_rest) * Ur + Uo
        Ub = float((w * U).sum())
        return float(np.sqrt((w * (U - Ub) ** 2).sum())) / abs(Ub), rho_rest

    lo, hi, step = scan
    grid = np.arange(lo, hi + 0.5 * step, step)
    vals = np.array([disp(g)[0] for g in grid])
    i = int(np.nanargmin(vals))
    rho_R = float(grid[i])
    d_min, rho_rest = disp(rho_R)
    d_uni, _ = disp(M_tot / (V_tot * 1e9))
    return {
        "rho_R": rho_R, "rho_rest": float(rho_rest),
        "contrast": float(rho_R / rho_rest),
        "improvement_pct": float(100 * (d_uni - d_min) / d_uni),
        "dispersion_uniform": float(d_uni), "dispersion_min": float(d_min),
        "excess_mass_pct": float(100 * (rho_R - rho_rest) * V_R * 1e9 / M_tot),
        "interior_minimum": bool(0 < i < len(grid) - 1),
        "facets_used": int(mask.sum()),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("obj")
    ap.add_argument("--mass", type=float, default=1.6495e15,
                    help="assumed total mass in kg; the default is the "
                         "400 kg/m3 case")
    ap.add_argument("--period", type=float, default=15.92)
    ap.add_argument("--scan", default="25,3000,5")
    ap.add_argument("--record", default="arrokoth_clean.json")
    a = ap.parse_args()
    scan = tuple(float(x) for x in a.scan.split(","))

    t0 = time.time()
    print(f"Loading {a.obj} ...")
    V, F = load_obj(a.obj)
    print(f"  {len(F)} facets")

    print("\nIdentifying buried facets ...")
    buried, groups = find_buried(V, F)
    keep = ~buried
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    print(f"  {len(groups)} components")
    print(f"  {buried.sum()} facets buried, "
          f"{100*A[buried].sum()/A.sum():.4f} per cent of the surface area")
    if buried.sum() == 0:
        print("  Nothing to remove; the two runs would be identical.")
        return 0

    V_tot = abs(de.closed_volume(V, F))
    print(f"  volume {V_tot:.2f} km^3, unchanged by the removal "
          f"(the polyhedron is untouched)")

    # the reference plane and the fraction sweep of the paper
    planes = [("neck", -4.7863)]
    for f in (0.400, 0.445, 0.495, 0.550, 0.612, 0.680, 0.756):
        lo, hi = float(V[:, 0].min()), float(V[:, 0].max())
        for _ in range(45):
            mid = 0.5 * (lo + hi)
            try:
                Vr, Fr = de.clip_halfspace(V, F, 0, mid, True)
                fr = abs(de.closed_volume(Vr, Fr)) / V_tot
            except Exception:
                fr = 0.0
            if fr > f:
                lo = mid
            else:
                hi = mid
        planes.append((f"frac {100*f:.1f}%", 0.5 * (lo + hi)))

    out = []
    print(f"\n{'plane':<12} {'x km':>9} | {'all facets':^26} | "
          f"{'buried removed':^26} | {'change':>9}")
    print(f"{'':<12} {'':>9} | {'rho_R':>7} {'contr':>8} {'impr%':>8} | "
          f"{'rho_R':>7} {'contr':>8} {'impr%':>8} | {'contrast':>9}")
    for lab, x in planes:
        try:
            ra = invert(V, F, np.ones(len(F), bool), a.mass, a.period, x, scan)
            rb = invert(V, F, keep, a.mass, a.period, x, scan)
        except Exception as e:
            print(f"{lab:<12} {x:>9.4f}   failed: {e}")
            continue
        dc = 100 * (rb["contrast"] / ra["contrast"] - 1)
        print(f"{lab:<12} {x:>9.4f} | {ra['rho_R']:>7.0f} "
              f"{ra['contrast']:>8.4f} {ra['improvement_pct']:>8.3f} | "
              f"{rb['rho_R']:>7.0f} {rb['contrast']:>8.4f} "
              f"{rb['improvement_pct']:>8.3f} | {dc:>+8.3f}%")
        out.append({"plane": lab, "x_split_km": x, "all": ra, "clean": rb,
                    "contrast_change_pct": dc})

    json.dump({"buried_facets": int(buried.sum()),
               "buried_area_pct": float(100 * A[buried].sum() / A.sum()),
               "components": len(groups), "results": out},
              open(a.record, "w"), indent=1)

    ch = [abs(r["contrast_change_pct"]) for r in out]
    step = scan[2]
    dr = max(abs(r["clean"]["rho_R"] - r["all"]["rho_R"]) for r in out)
    print(f"\n{'='*76}\nVERDICT\n{'='*76}")
    print(f"  largest change in contrast: {max(ch):.3f} per cent")
    print(f"  largest change in rho_R:    {dr:.1f} kg/m^3, against a scan "
          f"step of {step:.0f}")
    if dr < step and max(ch) < 1.0:
        print("\n  The buried facets do not affect the result. Every "
              "recovered density is")
        print("  identical to within one scan step and no contrast moves by "
              "one per cent.")
        print("  Report the fact and the measurement; no number in the paper "
              "needs changing.")
    else:
        print("\n  The buried facets DO affect the result. The values in the "
              "paper come from")
        print("  an average that includes points inside the body and should "
              "be replaced by")
        print("  the cleaned column above.")
    print(f"\n  written to {a.record}   [{(time.time()-t0)/60:.1f} min]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
