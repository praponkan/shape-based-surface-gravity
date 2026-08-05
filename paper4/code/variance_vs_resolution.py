#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the KEY hypothesis (Kanamaru 2019): the area-weighted potential variance
is LESS sensitive to shape-model resolution than the slope average.
If confirmed, potential variance is the robust metric for density estimation
(Paper V) and a headline finding for Paper IV.

Metrics per resolution (library gravity, uniform density):
  slope_avg   = area-weighted mean gravitational+rotational slope (deg)
  pot_std     = area-weighted std of surface geopotential, normalized by mean

USAGE: python variance_vs_resolution.py <obj> <density_gcc> <period_h>
"""
import sys, math, numpy as np
from surface_slope_lib import load_obj, decimate_cluster, outward_normals, centroids, volume
try:
    import polyhedral_gravity as pg
except ImportError:
    sys.exit("need polyhedral-gravity library")

def metrics(V,F,rho_gcc,period_hours):
    nf=outward_normals(V,F); C=centroids(V,F)
    p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    area=0.5*np.linalg.norm(np.cross(p1-p0,p2-p0),axis=1)   # km^2
    Vm=V*1000.0; Cm=C*1000.0
    rho_si=rho_gcc*1000.0
    ext=np.linalg.norm(Vm.max(0)-Vm.min(0)); eps=1e-4*ext
    pts=(Cm-eps*nf)
    poly=pg.Polyhedron(polyhedral_source=(Vm.tolist(),F.tolist()),
        density=rho_si,normal_orientation=pg.NormalOrientation.OUTWARDS,
        integrity_check=pg.PolyhedronIntegrity.DISABLE)
    res=pg.evaluate(poly,pts.tolist(),parallel=True)
    U=np.array([r[0] for r in res])          # potential
    g=np.array([r[1] for r in res])          # accel (inward)
    om=2*math.pi/(period_hours*3600.0)
    rp=Cm.copy(); rp[:,2]=0.0
    a_cf=om**2*rp                             # centrifugal outward (corrected)
    pot_cf=-0.5*om**2*(Cm[:,0]**2+Cm[:,1]**2)
    g_eff=g+a_cf; gmag=np.linalg.norm(g_eff,axis=1)
    cos=np.einsum('ij,ij->i',-g_eff,nf)/np.clip(gmag,1e-300,None)
    slope=np.degrees(np.arccos(np.clip(cos,-1,1)))
    Utot=-U+pot_cf   # physical potential (library U positive-convention)
    # area-weighted slope average
    slope_avg=np.sum(slope*area)/np.sum(area)
    # area-weighted potential mean & normalized std
    U_avg=np.sum(Utot*area)/np.sum(area)
    U_var=np.sum(((Utot-U_avg)**2)*area)/np.sum(area)
    pot_std_norm=math.sqrt(U_var)/abs(U_avg)   # dimensionless
    return slope_avg, pot_std_norm

obj=sys.argv[1]; rho=float(sys.argv[2]); P=float(sys.argv[3])
print(f"Loading {obj} ...")
V0,F0=load_obj(obj); print(f"  full facets {len(F0)}")
levels=[2000,5000,10000,20000,35000]
print(f"\n{'facets':>7} {'slope_avg':>10} {'pot_std_n':>11}")
rows=[]
for tgt in levels:
    V,F=decimate_cluster(V0,F0,tgt)
    sa,ps=metrics(V,F,rho,P)
    print(f"{len(F):>7} {sa:>10.3f} {ps:>11.5f}")
    rows.append((len(F),sa,ps))
# sensitivity = spread across resolutions, normalized by mean
sa_all=np.array([r[1] for r in rows]); ps_all=np.array([r[2] for r in rows])
print(f"\nsensitivity (max-min)/mean across resolutions:")
print(f"  slope_avg    : {(sa_all.max()-sa_all.min())/sa_all.mean()*100:.1f}%")
print(f"  pot_std_norm : {(ps_all.max()-ps_all.min())/ps_all.mean()*100:.1f}%")
print("\nIf pot_std_norm sensitivity < slope_avg sensitivity")
print("=> potential variance is the resolution-robust metric (Kanamaru confirmed).")
