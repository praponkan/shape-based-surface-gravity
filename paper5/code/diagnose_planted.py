#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Is the planted-anomaly failure a convergence problem or a real one?

    python diagnose_planted.py                  contrast 1.50, the default
    python diagnose_planted.py --contrast 1.25
    python diagnose_planted.py --iters 400 --gain 0.10

The first run of planted_anomaly.py returned contrasts below unity for every
planted contrast above it: 1.50 came back as 0.709, 2.00 as 0.572. Before that
can be read as a statement about identifiability, three features of the run
have to be explained, because each of them is equally consistent with the
experiment simply not having finished.

    Every case used all forty relaxation iterations. The convergence test never
    fired, so no surface had settled.

    The recovered region density was 1920 kg/m^3 for planted contrasts of 1.50,
    1.75 and 2.00 alike. An optimiser responding to its input does not return
    the same value three times from three different inputs.

    The residual dispersion saturated near 0.0335, and its correlation with the
    recovery error is -0.84. The worse the surface failed to relax, the worse
    the recovery: that is the signature of an unfinished experiment.

There is also a defect in the original script. The dividing plane was placed to
hold a chosen fraction of the STARTING volume, and the same plane was then used
on the RELAXED shape, whose volume distribution has changed. The inversion was
therefore asked about a different region from the one that was planted. This
script re-solves for the plane after relaxation so that the fraction matches.

What the outcome means
----------------------
    residual falls below about 0.01 and the recovered contrast climbs toward
    the planted one
        -> the earlier run had not converged. Rerun the ladder with more
           iterations and the result stands or falls on the new numbers.

    residual stays near 0.0335 however long it runs, and the recovered contrast
    stays below unity
        -> the surface cannot be relaxed onto an equipotential of the planted
           interior within this shape family, and the objective is minimised by
           a density other than the one that generated the surface. That is a
           genuine identifiability result and the most consequential finding in
           the series.

The two are distinguished by watching the residual, which is printed every ten
iterations.
"""

import argparse
import json
import sys
import time

import numpy as np

import density_estimate as de
import planted_anomaly as pa


def plane_for_fraction(V, F, frac):
    """Bisect for the plane leaving `frac` of the volume above it."""
    V_tot = abs(de.closed_volume(V, F))
    lo, hi = float(V[:, 0].min()), float(V[:, 0].max())
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        try:
            Vr, Fr = de.clip_halfspace(V, F, 0, mid, True)
            f = abs(de.closed_volume(Vr, Fr)) / V_tot
        except Exception:
            f = 0.0
        if f > frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def relax_verbose(V, F, rho_R, rho_rest, x_split, period_h, n_iter, gain,
                  report_every=10):
    """Relaxation with the history reported, and no early exit on a plateau.

    The original stopped when two successive dispersions agreed to 1e-9, which
    on a slowly creeping surface can fire long before the surface has settled.
    Here the full history is kept so that a plateau can be distinguished from
    slow progress by eye.
    """
    V = V.copy()
    vol0 = abs(de.closed_volume(V, F))
    hist = []
    for it in range(n_iter):
        U, _ = pa.geopotential(V, F, rho_R, rho_rest, x_split, period_h)
        p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        A = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        w = A / A.sum()
        Ubar = float((w * U).sum())
        disp = float(np.sqrt((w * (U - Ubar) ** 2).sum())) / abs(Ubar)
        hist.append(disp)
        if it % report_every == 0 or it == n_iter - 1:
            tail = ""
            if it >= 2 * report_every:
                drop = 100 * (hist[-1] / hist[-1 - report_every] - 1)
                tail = f"   change over the last {report_every}: {drop:+.3f}%"
            print(f"      iter {it:>4}  dispersion {disp:.7f}{tail}")

        res = (U - Ubar) / abs(Ubar)
        vres = np.zeros(len(V))
        vw = np.zeros(len(V))
        for k in range(3):
            np.add.at(vres, F[:, k], res * A)
            np.add.at(vw, F[:, k], A)
        vres /= np.maximum(vw, 1e-300)
        r = np.linalg.norm(V, axis=1)
        V = V * (1.0 - gain * vres)[:, None]
        V *= (vol0 / abs(de.closed_volume(V, F))) ** (1.0 / 3.0)
    return V, hist


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contrast", type=float, default=1.50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--gain", type=float, default=0.15)
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--period", type=float, default=12.13)
    ap.add_argument("--bulk", type=float, default=2000.0)
    ap.add_argument("--ntheta", type=int, default=120)
    ap.add_argument("--nphi", type=int, default=60)
    ap.add_argument("--record", default="diagnose_planted.json")
    a = ap.parse_args()

    t0 = time.time()
    print(f"DIAGNOSTIC: planted contrast {a.contrast:.2f}, "
          f"{a.iters} iterations at gain {a.gain}")
    print(f"The earlier run used 40 iterations at gain 0.35 and returned "
          f"{0.709 if abs(a.contrast-1.5) < 1e-9 else float('nan'):.3f} "
          f"with a residual of 0.0336.\n")

    V, F = pa.start_body(n_theta=a.ntheta, n_phi=a.nphi)
    V_tot = abs(de.closed_volume(V, F))
    M_tot = a.bulk * V_tot * 1e9
    print(f"  starting mesh: {len(F)} facets, volume {V_tot:.6f} km^3")

    x0 = plane_for_fraction(V, F, a.frac)
    V_R0 = abs(de.closed_volume(*de.clip_halfspace(V, F, 0, x0, True)))
    rho_rest = M_tot / ((a.contrast - 1.0) * V_R0 * 1e9 + V_tot * 1e9)
    rho_R = a.contrast * rho_rest
    print(f"  planted: rho_R {rho_R:.1f}, rho_rest {rho_rest:.1f}, "
          f"region {100*V_R0/V_tot:.2f}% at x = {x0:.5f}\n")

    print("  relaxing:")
    Vrel, hist = relax_verbose(V, F, rho_R, rho_rest, x0, a.period,
                               a.iters, a.gain)

    # --- the correction: re-solve the plane on the relaxed shape ---------
    x1 = plane_for_fraction(Vrel, F, a.frac)
    V_R1 = abs(de.closed_volume(*de.clip_halfspace(Vrel, F, 0, x1, True)))
    V_tot1 = abs(de.closed_volume(Vrel, F))
    print(f"\n  after relaxation the same fraction sits at x = {x1:.5f}, "
          f"not {x0:.5f}")
    print(f"  region at the OLD plane: "
          f"{100*abs(de.closed_volume(*de.clip_halfspace(Vrel, F, 0, x0, True)))/V_tot1:.2f}%")
    print(f"  region at the NEW plane: {100*V_R1/V_tot1:.2f}%")

    out = {"contrast_planted": a.contrast, "iters": a.iters, "gain": a.gain,
           "rho_R_planted": rho_R, "rho_rest_planted": rho_rest,
           "dispersion_history": hist,
           "x_split_before": x0, "x_split_after": x1}

    for lab, x in (("old plane", x0), ("re-solved plane", x1)):
        r = pa.invert(Vrel, F, M_tot, a.period, x, a.bulk)
        out[lab.replace(" ", "_")] = r
        print(f"\n  inversion at the {lab}: rho_R {r['rho_R_recovered']:.0f}, "
              f"rho_rest {r['rho_rest_recovered']:.0f}, contrast "
              f"{r['contrast_recovered']:.3f}, improvement "
              f"{r['improvement_pct']:.2f}%")

    json.dump(out, open(a.record, "w"), indent=1)

    # --- verdict ----------------------------------------------------------
    d0, dN = hist[0], hist[-1]
    last_tenth = hist[-max(a.iters // 10, 2):]
    creep = abs(last_tenth[-1] / last_tenth[0] - 1)
    best = out["re-solved_plane"]["contrast_recovered"]
    prev_resid = 0.0336

    print(f"\n{'='*72}\nVERDICT\n{'='*72}")
    print(f"  dispersion {d0:.6f} -> {dN:.6f}")
    print(f"  change over the final tenth of the run: {100*creep:.4f}%")
    print(f"  earlier run reached {prev_resid:.4f} in 40 iterations at "
          f"gain 0.35")
    print(f"  recovered contrast, re-solved plane: {best:.3f} against "
          f"{a.contrast:.3f} planted")
    print()

    settled = creep < 0.005
    improved = dN < 0.6 * prev_resid
    closer = abs(best - a.contrast) < 0.6 * abs(0.709 - a.contrast)

    if improved and closer:
        print("  The earlier run had not converged. The residual is "
              "substantially lower and the")
        print("  recovery substantially better. Rerun the full ladder with "
              "these settings before")
        print("  drawing any conclusion about identifiability.")
    elif settled and not improved:
        print("  The surface has settled and the residual has not improved. "
              "Within this shape")
        print("  family the surface cannot be brought onto an equipotential "
              "of the planted")
        print("  interior, and the objective is minimised by a density other "
              "than the one that")
        print("  generated the surface. If this survives a check on a "
              "different starting shape,")
        print("  it is a genuine identifiability result and the most "
              "consequential in the series.")
    elif not settled:
        print("  Still creeping: the surface has not settled even now. "
              "Increase --iters again,")
        print("  or lower --gain, before reading anything into the recovery.")
    else:
        print("  Mixed: the residual improved but the recovery did not, or "
              "the reverse. Report")
        print("  both and do not summarise until a second starting shape has "
              "been tried.")

    print(f"\n  written to {a.record}   [{(time.time()-t0)/60:.1f} min]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
