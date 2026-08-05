#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Locate where slope>90 facets are (neck/concave vs spread). Fast: decimates + progress."""
import sys, numpy as np
from surface_slope import load_obj, surface_fields, centroids, decimate_cluster
obj=sys.argv[1]; rho=float(sys.argv[2]); P=float(sys.argv[3])
target=10000
if '--target' in sys.argv: target=int(sys.argv[sys.argv.index('--target')+1])
V,F=load_obj(obj)
print(f"loaded facets {len(F)}")
if len(F)>target:
    V,F=decimate_cluster(V,F,target); print(f"decimated -> {len(F)}")
print("computing (progress every 1000)...")
s,gp,lat,gm=surface_fields(V,F,rho_gcc=rho,period_hours=P,progress=True)
C=centroids(V,F)
bad=s>90
print(f"\nfacets {len(F)}, slope>90: {bad.sum()} ({bad.mean()*100:.1f}%)")
print(f"mean slope {s.mean():.1f}  max {s.max():.1f}")
if bad.sum()>0:
    xb=C[bad,0]
    xr=C[:,0].max()-C[:,0].min()
    xmid=(C[:,0].max()+C[:,0].min())/2
    print(f"\nx range body: {C[:,0].min():.3f} to {C[:,0].max():.3f} (mid {xmid:.3f})")
    print(f"x of slope>90: {xb.min():.3f} to {xb.max():.3f}")
    near_mid=np.mean(np.abs(xb-xmid)<0.15*xr)
    print(f"fraction of slope>90 near center (neck): {near_mid*100:.0f}%")
    print("=> concentrated at neck = WS concave problem" if near_mid>0.5 else "=> spread out = other issue")
else:
    print("no slope>90 (decimation may hide neck - try higher --target)")
