#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test whether mean slope decreases as mesh is coarsened (toward the resolution
used by published slope maps, e.g. Eros 3-deg grid ~ 7000 facets).
If mean slope drops toward published as facets decrease, then facet-scale
surface roughness (not the framework) explains the high slope.

USAGE: python slope_vs_resolution.py <obj> <density_gcc> <period_h>
Runs several decimation levels with the LIBRARY solver.
"""
import sys, numpy as np, math
from surface_slope_lib import load_obj, decimate_cluster, surface_fields

obj=sys.argv[1]; rho=float(sys.argv[2]); P=float(sys.argv[3])
levels=[2000,4000,7000,15000,30000]
print(f"Loading {obj} ...")
V0,F0=load_obj(obj); print(f"  full facets {len(F0)}")
print(f"\n{'target':>7} {'actual':>7} {'mean':>7} {'median':>7} {'>30deg%':>8}")
for tgt in levels:
    V,F=decimate_cluster(V0,F0,tgt)
    s,gp,lat,gm=surface_fields(V,F,rho,P,progress=False)
    over30=(s>30).mean()*100
    print(f"{tgt:>7} {len(F):>7} {s.mean():>7.2f} {np.median(s):>7.2f} {over30:>8.1f}")
print("\nIf mean slope DECREASES toward published as facets decrease")
print("=> facet-scale roughness explains the offset (framework is correct).")
