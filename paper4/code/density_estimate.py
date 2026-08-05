#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper V pilot -- Interior density estimation by potential-variance minimization
(after Kanamaru & Sasaki 2019) for a two-lobe body (Itokawa Head/Body).

Method:
  1. Split the shape at a plane x = x_split into Head (x > x_split) and Body.
  2. Fix total mass M_total. For a trial rho_Head, solve rho_Body from
        rho_Head*V_Head + rho_Body*V_Body = M_total.
  3. Gravity is linear in density, so the field of a two-density body =
        rho_Head * field(Head-region, unit density)
      + rho_Body * field(Body-region, unit density).
     We realise each region as a closed polyhedron by capping at the split
     plane (approximation) OR, more simply here, use SUPERPOSITION of the full
     body at rho_Body PLUS the Head region at (rho_Head - rho_Body).
        field(total) = rho_Body*field(full,1) + (rho_Head-rho_Body)*field(headcap,1)
     This requires a closed Head sub-polyhedron. We build it by taking Head
     facets + a planar cap at x_split.
  4. Compute area-weighted surface potential variance; scan rho_Head; find min.
  5. Report rho_Head at minimum, implied rho_Body, and COM-COF offset.

USAGE: python density_estimate.py "Itokawa Hayabusa 50k poly.obj" --xsplit 0.150 --mtotal 3.58e10
Lengths in km. x_split in km (Itokawa Head boundary ~0.150 km).
"""
import sys, math, numpy as np
from surface_slope_lib import load_obj, outward_normals, centroids, decimate_cluster
try:
    import polyhedral_gravity as pg
except ImportError:
    sys.exit("need polyhedral-gravity library")

def tetra_volumes(V,F):
    """Signed tetra volume from origin for each facet (sums to body volume)."""
    p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    return np.einsum('ij,ij->i',p0,np.cross(p1,p2))/6.0  # km^3, signed

def region_volume_and_com(V,F,mask_facets):
    """Volume & COM of the sub-region using divergence (facets in mask + cap).
    Approximate: use signed tetra from origin over masked facets only.
    For a convex-ish cap this approximates the lobe volume."""
    tv=tetra_volumes(V,F)
    vol=tv[mask_facets].sum()
    # COM of region ~ weighted centroid of tetra (3/4 point ~ facet centroid*3/4)
    C=centroids(V,F)
    com=np.sum((C[mask_facets]*tv[mask_facets,None]),axis=0)/vol
    return abs(vol), com

def build_capped_region(V,F,side='head',x_split=0.150):
    """Return (Vr,Fr) closed polyhedron for the Head (x>split) or Body region,
    by keeping facets on that side and adding a planar cap at x=x_split.
    Simple cap: project boundary onto plane and fan-triangulate to centroid."""
    C=centroids(V,F)
    if side=='head': keep = C[:,0] > x_split
    else:            keep = C[:,0] <= x_split
    Fk=F[keep]
    # collect boundary vertices near the split (used facets' verts)
    used=np.unique(Fk)
    remap={old:i for i,old in enumerate(used)}
    Vr=V[used].copy()
    Fr=np.array([[remap[a] for a in f] for f in Fk])
    # planar cap: find vertices of kept facets that lie closest to split plane
    # approximate cap: add centroid-on-plane and fan to all boundary edges
    # (boundary edges = edges used by exactly one kept facet)
    from collections import defaultdict
    ec=defaultdict(list)
    for fi,f in enumerate(Fr):
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            ec[tuple(sorted((a,b)))].append(fi)
    boundary=[e for e,fs in ec.items() if len(fs)==1]
    if boundary:
        bverts=np.unique([v for e in boundary for v in e])
        cap_center=Vr[bverts].mean(0)
        cap_center[0]=x_split  # put on plane
        ci=len(Vr); Vr=np.vstack([Vr,cap_center])
        capF=[[e[0],e[1],ci] for e in boundary]
        Fr=np.vstack([Fr,capF])
    return Vr,Fr

def surf_potential(V,F,region_fields,rho_head,rho_body,period_hours):
    """potential at full-body surface via superposition:
       U = rho_body*U_full_unit + (rho_head-rho_body)*U_headcap_unit + U_cf"""
    (U_full,)=region_fields['full']
    (U_head,)=region_fields['head']
    Cm=region_fields['C']*1000.0
    om=2*math.pi/(period_hours*3600.0)
    pot_cf=-0.5*om**2*(Cm[:,0]**2+Cm[:,1]**2)
    U = -(rho_body*U_full + (rho_head-rho_body)*U_head) + pot_cf  # physical sign
    return U

def eval_potential_unit(Vr,Fr,pts_m):
    Vm=Vr*1000.0
    poly=pg.Polyhedron(polyhedral_source=(Vm.tolist(),Fr.tolist()),
        density=1.0, normal_orientation=pg.NormalOrientation.OUTWARDS,
        integrity_check=pg.PolyhedronIntegrity.DISABLE)
    res=pg.evaluate(poly,pts_m.tolist(),parallel=True)
    return np.array([r[0] for r in res])

if __name__=="__main__":
    obj=sys.argv[1]
    xsplit=float(sys.argv[sys.argv.index('--xsplit')+1]) if '--xsplit' in sys.argv else 0.150
    Mtot=float(sys.argv[sys.argv.index('--mtotal')+1]) if '--mtotal' in sys.argv else 3.58e10
    P=float(sys.argv[sys.argv.index('--period')+1]) if '--period' in sys.argv else 12.1324
    print(f"Loading {obj} ...")
    V,F=load_obj(obj)
    if len(F)>16000:
        V,F=decimate_cluster(V,F,16000); print(f"  decimated -> {len(F)}")
    # volumes (km^3)
    tv=tetra_volumes(V,F)
    C=centroids(V,F)
    head_mask=C[:,0]>xsplit
    V_head=abs(tv[head_mask].sum())
    V_total=abs(tv.sum())
    V_body=V_total-V_head
    print(f"  x_split={xsplit} km  V_total={V_total:.4e} V_head={V_head:.4e} V_body={V_body:.4e} km^3")
    # convert volumes km^3 -> m^3 for mass
    V_total_m=V_total*1e9; V_head_m=V_head*1e9; V_body_m=V_body*1e9
    # build head-cap region (closed) for superposition
    Vh,Fh=build_capped_region(V,F,'head',xsplit)
    print(f"  head-cap region facets {len(Fh)}")
    # field points = full-body surface centroids nudged inward
    nf=outward_normals(V,F)
    ext=np.linalg.norm((V.max(0)-V.min(0))*1000); eps=1e-4*ext
    pts_m=(C*1000.0 - eps*nf)
    p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    area=0.5*np.linalg.norm(np.cross(p1-p0,p2-p0),axis=1)
    print("  precomputing unit-density fields (full + head-cap)...")
    U_full=eval_potential_unit(V,F,pts_m)
    U_head=eval_potential_unit(Vh,Fh,pts_m)
    fields={'full':(U_full,),'head':(U_head,),'C':C}
    # scan rho_Head
    print(f"\n  {'rho_Head':>9} {'rho_Body':>9} {'pot_var_norm':>13}")
    best=None
    results=[]
    for rhoH in np.arange(1500,3600,50.0):  # kg/m^3
        rhoB=(Mtot - rhoH*V_head_m)/V_body_m
        if rhoB<=0: continue
        U=surf_potential(V,F,fields,rhoH,rhoB,P)
        Uavg=np.sum(U*area)/np.sum(area)
        Uvar=np.sum(((U-Uavg)**2)*area)/np.sum(area)
        pv=math.sqrt(Uvar)/abs(Uavg)
        results.append((rhoH,rhoB,pv))
        if best is None or pv<best[2]: best=(rhoH,rhoB,pv)
    for rhoH,rhoB,pv in results:
        mark=' <== MIN' if (rhoH,rhoB,pv)==best else ''
        if rhoH%200<50: print(f"  {rhoH:>9.0f} {rhoB:>9.0f} {pv:>13.6f}{mark}")
    print(f"\n  MINIMUM potential variance at:")
    print(f"    rho_Head = {best[0]:.0f} kg/m^3   (Kanamaru: 2750)")
    print(f"    rho_Body = {best[1]:.0f} kg/m^3   (Kanamaru: ~1930)")
    # COM-COF offset
    tvm=tv*1e9
    rhomap=np.where(head_mask,best[0],best[1])
    mass_el=np.abs(tvm)*rhomap
    COM=np.sum(C*mass_el[:,None],axis=0)/np.sum(mass_el)  # km
    COF=C.mean(0) if False else V.mean(0)  # center of figure ~ vertex mean
    offset_m=np.linalg.norm(COM-V.mean(0))*1000
    print(f"    COM-COF offset = {offset_m:.1f} m   (Kanamaru: ~16 m)")
