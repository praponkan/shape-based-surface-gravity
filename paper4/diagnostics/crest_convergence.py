#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crest convergence test: does equatorial-ridge-crest slope DECREASE as mesh
resolution rises? Distinguishes:
  (A) crest stays high  -> physical (density-heterogeneity signal, thesis holds)
  (B) crest drops toward published -> decimation noise (needs resolution, not thesis)

Runs several decimation levels and reports crest (|lat|<5) slope + poles.
USAGE: python crest_convergence.py Bennu_v20_200k.obj 1.194 4.296061
"""
import sys, numpy as np
from surface_slope import load_obj, decimate_cluster, surface_fields

obj=sys.argv[1]; rho=float(sys.argv[2]); P=float(sys.argv[3])
levels=[5000,10000,20000,30000]
print(f"Loading {obj} ...")
V0,F0=load_obj(obj); print(f"  full facets {len(F0)}")
print(f"\n{'target':>7} {'actual':>7} {'crest0-5':>9} {'peak(mid)':>10} {'pole75-90':>10}")
for tgt in levels:
    V,F=decimate_cluster(V0,F0,tgt)
    s,gp,lat,gm=surface_fields(V,F,rho_gcc=rho,period_hours=P,progress=False)
    crest=s[np.abs(lat)<5].mean()
    # find peak band
    bands=[(5,15),(15,25),(25,35),(35,45),(45,55)]
    peak=max(s[(np.abs(lat)>=lo)&(np.abs(lat)<hi)].mean() for lo,hi in bands)
    pole=s[np.abs(lat)>=75].mean()
    print(f"{tgt:>7} {len(F):>7} {crest:>9.2f} {peak:>10.2f} {pole:>10.2f}")
print("\nInterpretation:")
print("  crest DECREASING with resolution -> (B) decimation noise")
print("  crest STABLE (high) with resolution -> (A) density-heterogeneity signal")
