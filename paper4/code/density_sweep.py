#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper IV — Density sweep of Bennu surface slope.

KEY quantitative test (Scheeres et al.): mean surface slope falls as assumed
bulk density rises, because gravity scales with density while the centrifugal
term does not. Reproducing this from the shape model, an assumed density and
a specified spin state validates the framework
quantitatively (not just the spatial pattern).

Calibration approach (verified, unit-safe):
  - Compute raw WS acceleration once with rho=1 (arbitrary scale).
  - For each density rho: target GM = G * rho * V. Rescale the raw g by the
    single factor that makes the implied GM match, i.e. g_phys = g_raw * (GM/G)/V
    ... but since g_raw already equals (G * 1 * shape-integral), the physical
    field for density rho is simply g_phys = g_raw * rho_ratio where rho_ratio
    makes total mass = rho*V. We calibrate g so that GM_implied == G*rho*V by
    scaling to the WS mass integral. In practice: physical g = g_raw * rho_si,
    with rho in kg/km^3 and G in km^3 kg^-1 s^-2. We convert G to km-units so
    the whole computation is dimensionally consistent in km, s.

Units used throughout: length km, time s, mass kg.
  G = 6.674e-20 km^3 kg^-1 s^-2   (6.674e-11 m^3/kg/s^2 * (1e-3 km/m)^3)
  rho [kg/km^3] = rho[g/cm^3] * 1000 (kg/m^3 per g/cm^3) * 1e9 (m^3/km^3)
USAGE:
  python density_sweep.py Bennu_v20_200k.obj 4.296061
  python density_sweep.py Bennu_v20_200k.obj 4.296061 --target 30000 --rhos 0.85,1.0,1.15,1.194
"""
import sys, math
import numpy as np
from surface_slope import load_obj, onormals, centroids, decimate_cluster, _prep, _pa

G_KM = 6.674e-20   # km^3 kg^-1 s^-2

def volume_km3(V,F):
    p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    return abs(np.sum(np.einsum('ij,ij->i',p0,np.cross(p1,p2)))/6.0)

def run(V,F,rho_list,period_hours):
    N=onormals(V,F); C=centroids(V,F); prep=_prep(V,F)
    # raw WS accel with G_KM and rho=1 kg/km^3  (lengths already km)
    U_raw,g_raw=_pa(prep,C,G=G_KM,rho=1.0,progress=True)
    Vol=volume_km3(V,F)
    om=2*math.pi/(period_hours*3600.0)          # rad/s
    rp=C.copy(); rp[:,2]=0.0                     # km
    a_cf=-om**2*rp   # km/s^2; _pa returns OUTWARD-stored g, so centrifugal
                     # enters with a minus sign (verified against the analytic
                     # slope field of a uniformly rotating sphere)
    lat=np.degrees(np.arcsin(np.clip(C[:,2]/np.linalg.norm(C,axis=1),-1,1)))
    out={}
    for rho_gcc in rho_list:
        rho_kg_km3 = rho_gcc*1000.0*1e9          # kg/km^3
        g_phys = g_raw * rho_kg_km3              # km/s^2  (linear in rho)
        GM = G_KM*rho_kg_km3*Vol                 # km^3/s^2 (for reporting)
        g_eff=g_phys+a_cf
        gmag=np.linalg.norm(g_eff,axis=1)
        cos=np.einsum('ij,ij->i',g_eff,N)/np.clip(gmag,1e-300,None)
        slope=np.degrees(np.arccos(np.clip(cos,-1,1)))
        # centrifugal-to-gravity ratio at equator (diagnostic)
        eqmask=np.abs(lat)<20
        gg=np.linalg.norm(g_phys,axis=1)
        cf=np.linalg.norm(a_cf,axis=1)
        ratio=(cf[eqmask]/np.clip(gg[eqmask],1e-300,None)).mean()
        out[rho_gcc]=dict(mean=slope.mean(),median=float(np.median(slope)),
            eq=slope[eqmask].mean(),
            mid=slope[(np.abs(lat)>=20)&(np.abs(lat)<40)].mean(),
            pol=slope[np.abs(lat)>=60].mean() if (np.abs(lat)>=60).any() else float('nan'),
            GM=GM, cf_g_eq=ratio)
    return out

if __name__=="__main__":
    args=[a for a in sys.argv[1:] if not a.startswith('--')]
    obj=args[0]; P=float(args[1])
    target=30000
    if '--target' in sys.argv: target=int(sys.argv[sys.argv.index('--target')+1])
    rhos=[0.85,1.0,1.15,1.194]
    if '--rhos' in sys.argv:
        rhos=[float(x) for x in sys.argv[sys.argv.index('--rhos')+1].split(',')]
    print(f"Loading {obj} ...")
    V,F=load_obj(obj); print(f"  facets {len(F)}")
    if len(F)>target:
        V,F=decimate_cluster(V,F,target); print(f"  decimated -> {len(F)}")
    print(f"  volume {volume_km3(V,F):.6e} km^3, period {P} h")
    print(f"  computing WS once, sweeping rho {rhos} g/cm^3 ...")
    res=run(V,F,rhos,P)
    print("\n=== DENSITY SWEEP: mean slope vs assumed bulk density ===")
    print(f"  {'rho':>7} {'mean':>7} {'med':>7} {'eq<20':>7} {'mid':>7} {'pol':>7} {'cf/g_eq':>8} {'GM':>11}")
    for r in rhos:
        d=res[r]
        print(f"  {r:>7.3f} {d['mean']:>7.2f} {d['median']:>7.2f} {d['eq']:>7.2f} {d['mid']:>7.2f} {d['pol']:>7.2f} {d['cf_g_eq']:>8.3f} {d['GM']:>11.4e}")
    print("\n  Scheeres ref: rho 0.85->24deg, 1.15->15deg. Expect mean DECREASE w/ rho.")
    lo,hi=rhos[0],rhos[-1]
    print(f"  Our trend: {res[lo]['mean']:.2f}deg @rho{lo} -> {res[hi]['mean']:.2f}deg @rho{hi}")
