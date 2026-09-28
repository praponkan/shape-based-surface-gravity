#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Does the excess mass become the split-invariant quantity only when the signal
is strong?

    python synth_invariance.py [--record synth_invariance.json]

Background
----------
Across the sweep of split planes, Itokawa keeps its excess mass steady (half
range 5.1%) while the head density moves (13.1%). On 67P and Arrokoth the
order reverses: the density barely moves (3.5% and 0.6%) while the excess mass
swings by more than 40%. The proposed reason is that the compensation is the
signature of a real mass anomaly, and that Itokawa is the only one of the
three carrying a signal above the Paper IV noise band.

A further real body cannot decide this, because we cannot choose whether it
will carry a signal and we do not know its true interior. Here the geometry is
dialled instead: a family of contact binaries whose necks deepen while the
volume stays within 7%, so the signal strength varies and the threshold can be
located.

What is being tested
--------------------
The structure of the objective function, not the recovery of an imposed
density. Potential-dispersion minimization does not fit observations: it
assumes the true density is the one that flattens the surface geopotential.
Imposing an arbitrary density and asking for it back would fail by
construction and would test nothing.

Reported per body
-----------------
    improvement           how far the dispersion falls below the uniform case
    half-range of rho_R   spread of the recovered density over the sweep
    half-range of dM      spread of the recovered excess mass
    ratio                 dM spread divided by rho spread; below 1 means the
                          excess mass is the steadier quantity
"""

import json
import math
import sys

import numpy as np

import density_estimate as de
from synth_bodies import build_body, lobe_profile, closed_volume, is_watertight
from surface_slope_lib import outward_normals, centroids

RHO_BULK = 2000.0      # kg/m^3, Itokawa-like
PERIOD_H = 12.13
SCAN = (300.0, 4000.0, 25.0)
N_PLANES = 7
VR_SPAN = 1.90         # ratio of largest to smallest region volume across the
                       # sweep, matched to the Itokawa sweep so that bodies are
                       # compared over the same relative change of region size
                       # rather than the same distance in kilometres

# Two families, so the signal strength is spanned by design rather than by
# hope. A body with equal lobes is symmetric about its neck, so the
# dispersion-minimising contrast should be close to 1 and the signal weak. A
# body with unequal lobes has a genuine mass asymmetry to find. Within each
# family the neck is deepened while the volume stays within about 7%.
L_BIG = (0.15, -0.13, 0.20)
CONFIGS = [
    # near-spherical control, the synthetic analogue of Bennu
    ("sphere", [(0.15, 0.0, 0.16)]),

    # symmetric family: equal lobes, expected weak signal
    ("sym_waist", [(0.14, -0.11, 0.185), (0.14, 0.11, 0.185)]),
    ("sym_neck", [(0.14, -0.15, 0.185), (0.14, 0.15, 0.185)]),
    ("sym_deep", [(0.14, -0.18, 0.185), (0.14, 0.18, 0.185)]),

    # asymmetric family: unequal lobes, expected stronger signal
    ("asym_waist", [L_BIG, (0.12, 0.12, 0.17)]),
    ("asym_neck", [L_BIG, (0.12, 0.18, 0.17)]),
    ("asym_deep", [L_BIG, (0.12, 0.22, 0.17)]),
    ("asym_extreme", [(0.15, -0.13, 0.20), (0.10, 0.20, 0.14)]),
]


def analytic_neck(lobes):
    xs = np.linspace(min(c - a for _, c, a in lobes),
                     max(c + a for _, c, a in lobes), 2000)
    r = lobe_profile(xs, lobes)
    inner = (xs > xs.min() + 0.12 * np.ptp(xs)) & (xs < xs.max() - 0.12 * np.ptp(xs))
    ri, xi = r[inner], xs[inner]
    j = int(np.argmin(ri))
    local = 0 < j < len(ri) - 1
    return (float(xi[j]) if local else 0.0), float(ri[j] / max(A for A, _, _ in lobes)), local


def half_range_pct(v):
    """Half range as a percentage of the mean.

    Meaningless when the mean sits inside the spread, which happens whenever
    the quantity changes sign across the sweep; callers must check
    `crosses_zero` before using it.
    """
    v = np.asarray(v, float)
    m = abs(v.mean())
    return float(100 * (v.max() - v.min()) / 2 / m) if m > 0 else float("nan")


def crosses_zero(v):
    v = np.asarray(v, float)
    return bool(v.min() * v.max() < 0)


def scaling_slope(V_R, drho):
    """Slope of log(rho_R - rho_rest) against log(V_R).

    Since the excess mass is dM = drho * V_R, a slope of -1 means dM is held
    fixed as the plane moves, and a slope of 0 means the density difference is
    held fixed instead. Returns None if drho changes sign, in which case there
    is no anomaly to hold fixed and the question does not arise.
    """
    V_R = np.asarray(V_R, float)
    drho = np.asarray(drho, float)
    if drho.min() * drho.max() <= 0:
        return None
    return float(np.polyfit(np.log(V_R), np.log(np.abs(drho)), 1)[0])


def plane_for_fraction(V, F, V_tot, target, xlo, xhi, iters=30):
    """Bisect for the plane that leaves `target` of the volume above it."""
    for _ in range(iters):
        xm = 0.5 * (xlo + xhi)
        try:
            Vr, Fr = de.clip_halfspace(V, F, 0, xm, True)
            f = abs(closed_volume(Vr, Fr)) / V_tot
        except ValueError:
            f = 0.0
        if f > target:
            xlo = xm
        else:
            xhi = xm
    return 0.5 * (xlo + xhi)


def run_body(name, lobes, tol_rel=None):
    V, F = build_body(lobes)
    wt, dup, unm = is_watertight(F)
    src_ok, src_mf = de.check_manifold(F)
    if not src_ok:
        print(f"  WARNING: the source mesh for {name} is not a closed "
              f"2-manifold (chi = {src_mf['euler_characteristic']})")
    V_tot = abs(closed_volume(V, F))
    M_total = RHO_BULK * V_tot * 1e9

    x_neck, neck_ratio, has_neck = analytic_neck(lobes)

    # detector on the meshed body, as a side check
    O = V.mean(0)
    mid, prof = de.cross_section_profile(V, F, O)
    xd, wd, rd, ndiag = de.find_neck(mid, prof)

    nf = outward_normals(V, F)
    C = centroids(V, F)
    Cm = C * 1000.0
    ext = np.linalg.norm((V.max(0) - V.min(0)) * 1000.0)
    pts = Cm - 1e-4 * ext * nf
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    om = 2 * math.pi / (PERIOD_H * 3600.0)
    pot_rot = -0.5 * om ** 2 * (Cm[:, 0] ** 2 + Cm[:, 1] ** 2)

    U_full = de.unit_potential(V, F, pts)          # once per body
    d_uniform = de.dispersion(RHO_BULK * U_full, pot_rot, area)

    lo, hi, step = SCAN

    # Choose planes by the fraction of volume they leave above them, spanning
    # VR_SPAN in ratio and centred on the fraction at the neck. This makes the
    # sweep comparable across bodies of different shape, and comparable with
    # the Itokawa sweep, which spanned a factor 1.90 in region volume.
    Vn, Fn = de.clip_halfspace(V, F, 0, x_neck, True, tol_rel=tol_rel)
    f_neck = abs(closed_volume(Vn, Fn)) / V_tot
    r = math.sqrt(VR_SPAN)
    targets = np.geomspace(f_neck / r, f_neck * r, N_PLANES)
    x0, x1 = float(V[:, 0].min()), float(V[:, 0].max())
    planes = [plane_for_fraction(V, F, V_tot, float(t), x0, x1)
              for t in targets[::-1]]
    out = []
    n_nonmanifold = 0
    for xs_ in planes:
        try:
            Vr, Fr = de.clip_halfspace(V, F, 0, float(xs_), True,
                                       tol_rel=tol_rel)
        except ValueError:
            continue
        mf_ok, mf = de.check_manifold(Fr)
        if not mf_ok:
            print(f"    plane {xs_:+.5f}: region mesh is not a closed "
                  f"2-manifold (chi = {mf['euler_characteristic']}, "
                  f"{mf['unmatched_directed_edges']} unmatched edges); "
                  f"skipped")
            n_nonmanifold += 1
            continue

        V_R = abs(closed_volume(Vr, Fr))
        V_rest = V_tot - V_R
        if V_R <= 0 or V_rest <= 0:
            continue
        frac = 100 * V_R / V_tot
        if frac < 2 or frac > 98:
            continue
        cen_R = de.closed_centroid(Vr, Fr)
        U_R = de.unit_potential(Vr, Fr, pts)
        if not np.isfinite(U_R).all():
            continue
        rows = []
        for rho_R in np.arange(lo, hi + 0.5 * step, step):
            rho_rest = (M_total - rho_R * V_R * 1e9) / (V_rest * 1e9)
            if rho_rest <= 0:
                continue
            U = rho_rest * U_full + (rho_R - rho_rest) * U_R
            rows.append((rho_R, rho_rest, de.dispersion(U, pot_rot, area)))
        arr = np.array(rows)
        if len(arr) < 5 or not np.isfinite(arr[:, 2]).any():
            continue
        k = int(np.nanargmin(arr[:, 2]))
        interior = 0 < k < len(arr) - 1
        rho_R, rho_rest, dmin = float(arr[k, 0]), float(arr[k, 1]), float(arr[k, 2])
        excess = (rho_R - rho_rest) * V_R * 1e9
        com = excess * (cen_R[0] - de.closed_centroid(V, F)[0]) / M_total * 1000.0
        out.append({
            "x_split_km": float(xs_),
            "volume_fraction_pct": float(frac),
            "rho_R": rho_R, "rho_rest": rho_rest,
            "contrast": rho_R / rho_rest,
            "dispersion": dmin,
            "improvement_pct": 100 * (d_uniform - dmin) / d_uniform,
            "interior_minimum": bool(interior),
            "excess_mass_kg": float(excess),
            "excess_mass_pct_of_total": float(100 * excess / M_total),
            "com_offset_m": float(com),
            "sigma_rho_at_0p60pct": de.curvature_sigma(arr, k, 0.0060),
        })

    good = [p for p in out if p["interior_minimum"]]
    drho = [p["rho_R"] - p["rho_rest"] for p in good]
    VRs = [p["volume_fraction_pct"] for p in good]
    summary = {
        "name": name, "lobes": lobes,
        "watertight": bool(wt),
        "source_mesh_manifold": bool(src_ok),
        "source_euler_characteristic": src_mf["euler_characteristic"],
        "n_planes_rejected_nonmanifold": n_nonmanifold,
        "volume_km3": float(V_tot),
        "bulk_density": RHO_BULK, "M_total_kg": float(M_total),
        "period_h": PERIOD_H,
        "clip_tol_rel": (tol_rel if tol_rel is not None else 1e-10),
        "neck_ratio": neck_ratio,
        "x_neck_analytic": x_neck,
        "has_true_saddle": bool(has_neck),
        "detector_x_neck": (float(xd) if xd is not None else None),
        "detector_prominence": (ndiag["selected"]["prominence"]
                                if xd is not None else None),
        "uniform_dispersion": float(d_uniform),
        "n_planes_used": len(good),
        "fraction_at_neck": float(f_neck),
        "planes": out,
    }
    if len(good) >= 3:
        summary["improvement_max_pct"] = max(p["improvement_pct"] for p in good)
        summary["improvement_min_pct"] = min(p["improvement_pct"] for p in good)
        summary["rho_half_range_pct"] = half_range_pct([p["rho_R"] for p in good])
        summary["dM_half_range_pct"] = half_range_pct(
            [p["excess_mass_pct_of_total"] for p in good])
        summary["com_half_range_pct"] = half_range_pct(
            [p["com_offset_m"] for p in good])
        summary["dM_crosses_zero"] = crosses_zero(
            [p["excess_mass_pct_of_total"] for p in good])
        summary["drho_crosses_zero"] = crosses_zero(drho)
        summary["V_R_span_ratio"] = float(max(VRs) / min(VRs))
        summary["scaling_slope"] = scaling_slope(VRs, drho)
        s_ = summary["scaling_slope"]
        summary["interpretation"] = (
            "no anomaly: the density difference changes sign, so neither "
            "quantity has anything to hold fixed" if s_ is None else
            ("excess mass held fixed" if abs(s_ + 1) < 0.3 else
             ("density difference held fixed" if abs(s_) < 0.3 else
              f"intermediate, slope {s_:.2f}")))
        summary["invariance_ratio_dM_over_rho"] = (
            summary["dM_half_range_pct"] / summary["rho_half_range_pct"]
            if summary["rho_half_range_pct"] > 0 else float("inf"))
        summary["ratio_is_meaningful"] = not summary["dM_crosses_zero"]
    return summary


def main():
    argv = sys.argv[1:]
    rec = argv[argv.index("--record") + 1] if "--record" in argv else \
        "synth_invariance.json"
    tol_rel = (float(argv[argv.index("--cliptol") + 1])
               if "--cliptol" in argv else None)
    only = argv[argv.index("--only") + 1] if "--only" in argv else None

    results = []
    for name, lobes in CONFIGS:
        if only and only not in name:
            continue
        print(f"\n=== {name} ===")
        s = run_body(name, lobes, tol_rel=tol_rel)
        results.append(s)
        print(f"  volume {s['volume_km3']:.6f} km^3, watertight "
              f"{s['watertight']}, source chi = "
              f"{s['source_euler_characteristic']}, "
              f"neck/hump {s['neck_ratio']:.3f}, saddle "
              f"{s['has_true_saddle']}")
        if s["n_planes_rejected_nonmanifold"]:
            print(f"  {s['n_planes_rejected_nonmanifold']} planes rejected "
                  f"for non-manifold regions")
        print(f"  detector: x_neck {s['detector_x_neck']}, "
              f"prominence {s['detector_prominence']}")
        print(f"  {'x_split':>9} {'reg%':>6} {'rho_R':>7} {'rest':>7} "
              f"{'contr':>6} {'impr%':>8} {'dM/M%':>8} {'COM m':>8} {'int':>4}")
        for p in s["planes"]:
            print(f"  {p['x_split_km']:>9.4f} {p['volume_fraction_pct']:>6.1f} "
                  f"{p['rho_R']:>7.0f} {p['rho_rest']:>7.0f} "
                  f"{p['contrast']:>6.3f} {p['improvement_pct']:>8.3f} "
                  f"{p['excess_mass_pct_of_total']:>8.3f} "
                  f"{p['com_offset_m']:>8.2f} "
                  f"{'yes' if p['interior_minimum'] else 'EDGE':>4}")
        if "scaling_slope" in s:
            print(f"  improvement {s['improvement_min_pct']:.3f} to "
                  f"{s['improvement_max_pct']:.3f}%   "
                  f"V_R span {s['V_R_span_ratio']:.2f}")
            sl = s["scaling_slope"]
            print(f"  scaling slope log(drho) vs log(V_R): "
                  f"{'none, drho changes sign' if sl is None else f'{sl:+.2f}'}"
                  f"   ({s['interpretation']})")
            if s["ratio_is_meaningful"]:
                print(f"  half-range: rho {s['rho_half_range_pct']:.1f}%, "
                      f"dM {s['dM_half_range_pct']:.1f}%")
            else:
                print("  half-range of dM not reported: dM changes sign across "
                      "the sweep, so it has no meaningful mean")

    print(f"\n{'='*100}\nSUMMARY\n{'='*100}")
    print(f"{'body':<16} {'impr%':>8} {'V_R span':>9} {'slope':>8} "
          f"{'rho hr%':>9} {'dM hr%':>9} {'interpretation':<45}")
    ok = [s for s in results if "scaling_slope" in s]
    for s in ok:
        sl = s["scaling_slope"]
        dm = (f"{s['dM_half_range_pct']:.1f}" if s["ratio_is_meaningful"]
              else "sign flip")
        print(f"{s['name']:<16} {s['improvement_max_pct']:>8.2f} "
              f"{s['V_R_span_ratio']:>9.2f} "
              f"{('  --  ' if sl is None else f'{sl:+.2f}'):>8} "
              f"{s['rho_half_range_pct']:>9.1f} {dm:>9} "
              f"{s['interpretation']:<45}")
    print()
    print("  Itokawa, for comparison: improvement up to 26.4%, V_R span 1.90, "
          "slope -1.14")

    withslope = [s for s in ok if s["scaling_slope"] is not None]
    if len(withslope) >= 3:
        x = np.array([s["improvement_max_pct"] for s in withslope])
        y = np.array([s["scaling_slope"] for s in withslope])
        o = np.argsort(x)
        print("\nscaling slope against signal strength (sorted):")
        for a, b in zip(x[o], y[o]):
            print(f"  improvement {a:7.2f}%  ->  slope {b:+6.2f}"
                  f"   {'<- excess mass held fixed' if abs(b+1) < 0.3 else ''}")
        r = np.corrcoef(x, y)[0, 1]
        print(f"\n  correlation between signal strength and slope: {r:+.2f}")
        print("  a slope approaching -1 as the signal strengthens would support "
              "the proposed mechanism;")
        print("  no relation would refute it.")
    nosl = [s["name"] for s in ok if s["scaling_slope"] is None]
    if nosl:
        print(f"\n  bodies with no anomaly to hold fixed (density difference "
              f"changes sign): {', '.join(nosl)}")

    bad_src = [s["name"] for s in results
               if not s.get("source_mesh_manifold", True)]
    bad_reg = sum(s.get("n_planes_rejected_nonmanifold", 0) for s in results)
    print(f"\nmanifold audit: "
          f"{len(results) - len(bad_src)} of {len(results)} source meshes "
          f"closed, {bad_reg} clipped regions rejected")
    if bad_src:
        print(f"  non-manifold source meshes: {', '.join(bad_src)}")

    with open(rec, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)
    print(f"written to {rec}")


if __name__ == "__main__":
    main()
