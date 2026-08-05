#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ryugu facet-by-facet validation against JAXA ground truth
(Gravity_SHAPE_SFM_49k_v20180804_7.63_1200a.txt, Watanabe et al. 2019).

JAXA file provides per-plate GeopotentialSlope and Tpotential computed with
GFandSlope (Werner-Scheeres) at density 1200 kg/m^3, period 7.63 h.

We compute the same on the SAME 49k shape (NO decimation) and match plate-by-
plate by PlateID order (both use the same 49k plate ordering), then report:
  - slope: mean diff, RMS diff, Pearson r
  - potential: correlation
This is the strongest possible validation (same shape/method/density/period).

USAGE: python ryugu_facet_validation.py <ryugu_obj> <jaxa_txt>
  e.g. python ryugu_facet_validation.py "Ryugu_SHAPE_SFM_49k_v20180804.obj" "Gravity_SHAPE_SFM_49k_v20180804_7.63_1200a.txt"
IMPORTANT: run on FULL mesh (no decimation) so plate IDs line up 1:1.
"""
import sys, math, numpy as np
from surface_slope_lib import load_obj, outward_normals, centroids
try:
    import polyhedral_gravity as pg
except ImportError:
    sys.exit("need polyhedral-gravity library")

obj=sys.argv[1]; jaxa=sys.argv[2]
rho_gcc=1.2; period_h=7.63

# --- load JAXA ground truth ---
J=[]
with open(jaxa) as f:
    for line in f:
        if line.startswith('#') or not line.strip(): continue
        p=line.split()
        if len(p)<20: continue
        J.append((int(p[0]),float(p[1]),float(p[2]),float(p[3]),
                  float(p[14]),float(p[15]),float(p[19])))  # id,cx,cy,cz,Tpot,slope,area
J=np.array(J)
Jid=J[:,0].astype(int); Jc=J[:,1:4]; Jpot=J[:,4]; Jslope=J[:,5]; Jarea=J[:,6]
print(f"JAXA: {len(J)} plates, slope mean {Jslope.mean():.2f}")

# --- load shape, compute our slope on FULL mesh ---
V,F=load_obj(obj)
print(f"our shape: {len(F)} facets (must equal 49152 and match plate order)")
if len(F)!=len(J):
    print(f"WARNING: facet count {len(F)} != JAXA {len(J)}; matching by nearest centroid instead of index")

nf=outward_normals(V,F); C=centroids(V,F)
Vm=V*1000.0; Cm=C*1000.0
rho_si=rho_gcc*1000.0
ext=np.linalg.norm(Vm.max(0)-Vm.min(0)); eps=1e-4*ext
pts=(Cm-eps*nf)
poly=pg.Polyhedron(polyhedral_source=(Vm.tolist(),F.tolist()),density=rho_si,
    normal_orientation=pg.NormalOrientation.OUTWARDS,
    integrity_check=pg.PolyhedronIntegrity.DISABLE)
res=pg.evaluate(poly,pts.tolist(),parallel=True)
U=np.array([r[0] for r in res]); g=np.array([r[1] for r in res])
om=2*math.pi/(period_h*3600.0); rp=Cm.copy(); rp[:,2]=0
a_cf=om**2*rp; pot_cf=-0.5*om**2*(Cm[:,0]**2+Cm[:,1]**2)
g_eff=g+a_cf; gmag=np.linalg.norm(g_eff,axis=1)
cos=np.einsum('ij,ij->i',-g_eff,nf)/np.clip(gmag,1e-300,None)
Oslope=np.degrees(np.arccos(np.clip(cos,-1,1)))
Opot=-U+pot_cf

# --- match facets ---
if len(F)==len(J):
    # assume same plate ordering (JAXA .dat == .obj). verify by centroid distance.
    d=np.linalg.norm(C-Jc,axis=1)
    print(f"index-order centroid match: median dist {np.median(d)*1000:.2f} m, max {d.max()*1000:.1f} m")
    if np.median(d) > 0.005:  # >5 m median -> ordering differs, use NN
        print("  ordering differs -> nearest-neighbour match")
        from scipy.spatial import cKDTree
        tree=cKDTree(Jc); dist,idx=tree.query(C)
        Js=Jslope[idx]; Jp=Jpot[idx]
    else:
        Js=Jslope; Jp=Jpot
else:
    from scipy.spatial import cKDTree
    tree=cKDTree(Jc); dist,idx=tree.query(C)
    Js=Jslope[idx]; Jp=Jpot[idx]

# --- compare slope ---
diff=Oslope-Js
print(f"\n=== SLOPE facet-by-facet (ours vs JAXA) ===")
print(f"  our mean {Oslope.mean():.2f}  JAXA mean {Js.mean():.2f}")
print(f"  mean diff {diff.mean():+.2f}  RMS diff {np.sqrt((diff**2).mean()):.2f}  median|diff| {np.median(np.abs(diff)):.2f}")
r=np.corrcoef(Oslope,Js)[0,1]
print(f"  Pearson r = {r:.4f}")
print(f"  within 2 deg: {(np.abs(diff)<2).mean()*100:.1f}%   within 5 deg: {(np.abs(diff)<5).mean()*100:.1f}%")
# scatter analysis: is the diff a constant offset (bias) or random scatter?
print(f"  after removing mean bias: RMS {np.sqrt(((diff-diff.mean())**2).mean()):.2f} deg")
print(f"  slope where |diff|>10: {(np.abs(diff)>10).sum()} facets ({(np.abs(diff)>10).mean()*100:.1f}%)")
# correlation on low-slope facets (flat areas, less roughness-sensitive)
flat=Js<15
if flat.sum()>100:
    rf=np.corrcoef(Oslope[flat],Js[flat])[0,1]
    print(f"  r on flat facets (JAXA<15deg): {rf:.4f}")
# potential correlation
# JAXA potential uses opposite sign convention (Gpot>0, our U<0)
rp_pos=np.corrcoef(Opot,Jp)[0,1]
rp_neg=np.corrcoef(-Opot,Jp)[0,1]
rp_=max(abs(rp_pos),abs(rp_neg))
print(f"\n=== POTENTIAL (sign-convention aware) ===")
print(f"  Pearson |r| = {rp_:.4f}  (JAXA uses opposite sign; magnitude is what matters)")
print("\n=> high r (>0.95) + small RMS confirms per-facet framework correctness.")
