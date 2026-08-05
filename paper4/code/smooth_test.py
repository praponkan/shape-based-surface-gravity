#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test whether SMOOTHING the surface (Laplacian vertex smoothing, which actually
reduces facet-scale roughness, unlike decimation) lowers the mean slope toward
published values. If yes, the framework is correct and the offset is surface
roughness that published low-resolution/smoothed models do not contain.

USAGE: python smooth_test.py <obj> <density_gcc> <period_h>
"""
import sys, numpy as np
from collections import defaultdict
from surface_slope_lib import load_obj, surface_fields, decimate_cluster

def laplacian_smooth(V,F,iterations=1,factor=0.5):
    """Move each vertex toward the average of its neighbours (reduces roughness)."""
    # build vertex adjacency
    nbr=defaultdict(set)
    for f in F:
        a,b,c=f
        nbr[a]|={b,c}; nbr[b]|={a,c}; nbr[c]|={a,b}
    Vs=V.copy()
    for _ in range(iterations):
        newV=Vs.copy()
        for i in range(len(Vs)):
            if nbr[i]:
                avg=Vs[list(nbr[i])].mean(0)
                newV[i]=Vs[i]+factor*(avg-Vs[i])
        Vs=newV
    return Vs

obj=sys.argv[1]; rho=float(sys.argv[2]); P=float(sys.argv[3])
print(f"Loading {obj} ...")
V,F=load_obj(obj)
if len(F)>30000:
    V,F=decimate_cluster(V,F,30000); print(f"  decimated -> {len(F)} for speed")
print(f"\n{'smoothing':>12} {'mean':>7} {'median':>7} {'>30%':>6}")
for it in [0,1,3,5,10]:
    Vs=laplacian_smooth(V,F,iterations=it) if it>0 else V
    s,gp,lat,gm=surface_fields(Vs,F,rho,P,progress=False)
    print(f"{it:>12} {s.mean():>7.2f} {np.median(s):>7.2f} {(s>30).mean()*100:>6.1f}")
print("\nIf mean slope DROPS toward published with smoothing")
print("=> surface roughness explains the offset; framework is correct.")
