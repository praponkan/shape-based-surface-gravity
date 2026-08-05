#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROTATING-SPHERE VERIFICATION of the surface-slope field.

Why this test exists
--------------------
A NON-rotating sphere returns zero slope for EITHER sign of the centrifugal
term, so it cannot detect a sign error in the rotating frame. A uniformly
rotating sphere can: its slope field has a closed form, and the two sign
conventions give clearly different answers.

Analytic result
---------------
For a homogeneous sphere of radius R, surface gravity g = GM/R^2, spin rate w,
and k = w^2 R / g, the gravitational slope at latitude phi is

    cos(theta) = (1 - k cos^2 phi) / sqrt( cos^2 phi (1-k)^2 + sin^2 phi )

with the CORRECT convention  a_eff = A + w^2 r_perp   (A = attraction, inward;
centrifugal outward). Using the wrong sign is equivalent to replacing k by -k,
which at k = 0.5 changes the slope at latitude 45 deg from 18.4 to 11.3 deg.

Usage
-----
    python rotsphere_check.py            # pure-numpy implementation
    python rotsphere_check.py --lib      # polyhedral-gravity library implementation
"""
import sys, math
import numpy as np


def theta_analytic(phi_deg, k):
    """Closed-form gravitational slope of a uniformly rotating homogeneous sphere."""
    p = math.radians(phi_deg)
    c, s = math.cos(p), math.sin(p)
    num = 1.0 - k * c * c
    den = math.sqrt(c * c * (1.0 - k) ** 2 + s * s)
    return math.degrees(math.acos(np.clip(num / den, -1.0, 1.0)))


def main():
    use_lib = "--lib" in sys.argv
    if use_lib:
        from surface_slope_lib import surface_fields, make_icosphere as icosphere
    else:
        from surface_slope import surface_fields, icosphere

    G_KM = 6.674e-20          # km^3 kg^-1 s^-2
    R = 0.25                  # km
    rho_gcc = 1.95
    rho = rho_gcc * 1e12      # kg/km^3

    g = (4.0 / 3.0) * math.pi * G_KM * rho * R      # surface gravity, km/s^2
    k = 0.5                                          # target w^2 R / g
    om = math.sqrt(k * g / R)
    period_h = 2.0 * math.pi / om / 3600.0

    print(f"rotating-sphere check  ({'library' if use_lib else 'pure numpy'})")
    print(f"  R = {R} km, rho = {rho_gcc} g/cm^3, g = {g:.4e} km/s^2")
    print(f"  k = w^2 R / g = {k},  period = {period_h:.3f} h")

    V, F = icosphere(4, R)
    slope, geopot, lat, gmag = surface_fields(
        V, F, rho_gcc=rho_gcc, period_hours=period_h)

    print(f"\n  {'lat':>5} {'code':>9} {'analytic':>10} {'wrong sign':>12} {'err':>8}")
    ok = True
    for lo, hi in [(25, 35), (40, 50), (55, 65)]:
        m = (np.abs(lat) >= lo) & (np.abs(lat) < hi)
        mid = 0.5 * (lo + hi)
        got = slope[m].mean()
        want = theta_analytic(mid, k)
        wrong = theta_analytic(mid, -k)
        err = got - want
        if abs(err) > 1.0:
            ok = False
        print(f"  {mid:>5.0f} {got:>9.2f} {want:>10.2f} {wrong:>12.2f} {err:>+8.2f}")

    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'} "
          f"({'matches analytic' if ok else 'does NOT match analytic - check centrifugal sign'})")

    # a non-rotating sphere must also give ~0 (necessary but not sufficient)
    s0, _, _, _ = surface_fields(V, F, rho_gcc=rho_gcc, period_hours=1e9)
    print(f"  no-spin control: mean slope {s0.mean():.3f} deg (must be ~0)")


if __name__ == "__main__":
    main()
