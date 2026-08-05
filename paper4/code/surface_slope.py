#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper IV pilot — Surface slope & geopotential per facet of a small body.
Self-contained Werner-Scheeres (1996) polyhedron gravity + centrifugal term.
Verified on sphere (slope~0, |g|=GM/r^2) and oblate ellipsoid.

NEW: optional mesh decimation for large meshes (e.g. Bennu 200k) so the pilot
     runs in minutes. Uses vertex-clustering decimation (pure numpy, no deps).

USAGE:
    python surface_slope.py                                  # self-test
    python surface_slope.py Bennu_v20_200k.obj 1.194 4.296061   # obj, density(g/cm^3), period(h)
    python surface_slope.py Bennu_v20_200k.obj 1.194 4.296061   # obj, density(g/cm^3), period(h) --target 8000
"""
import sys, math
import numpy as np
from collections import defaultdict

# ----------------------------------------------------------------- mesh IO
def load_obj(path):
    # read raw bytes, normalise ALL line endings (\r\n, \r, \n) -> \n
    with open(path,'rb') as fh:
        raw=fh.read()
    text=raw.decode('utf-8',errors='ignore')
    text=text.replace('\r\n','\n').replace('\r','\n')
    V,F=[],[]
    for ln in text.split('\n'):
        s=ln.strip()
        if not s: continue
        t=s.split()
        if t[0]=='v' and len(t)>=4:
            try: V.append([float(t[1]),float(t[2]),float(t[3])])
            except ValueError: pass
        elif t[0]=='f' and len(t)>=4:
            try:
                idx=[int(tok.split('/')[0])-1 for tok in t[1:]]
                for k in range(1,len(idx)-1): F.append([idx[0],idx[k],idx[k+1]])
            except ValueError: pass
    return np.asarray(V,float),np.asarray(F,int)

def onormals(V,F):
    p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    n=np.cross(p1-p0,p2-p0);n/=np.linalg.norm(n,axis=1)[:,None]
    ctr=V.mean(0);fc=(p0+p1+p2)/3
    s=np.sign(np.einsum('ij,ij->i',n,fc-ctr));s[s==0]=1
    return n*s[:,None]

def centroids(V,F): return (V[F[:,0]]+V[F[:,1]]+V[F[:,2]])/3

# ----------------------------------------------------------------- decimation
def decimate_cluster(V,F,target_faces):
    """Vertex-clustering decimation: grid the bounding box, snap each vertex to
    its cell's representative, drop degenerate faces. Reduces facet count while
    preserving global shape (good enough for global slope statistics)."""
    if len(F)<=target_faces: return V,F
    # choose grid so that #occupied cells ~ target vertices (~ target_faces/2)
    tgt_v=max(target_faces//2, 200)
    mn,mx=V.min(0),V.max(0); ext=mx-mn; ext[ext==0]=1e-9
    # start with a guess and adjust grid resolution to hit target
    for gres in [16,20,24,28,32,40,48,56,64,80,96,128,160,200,256]:
        cell=ext/gres
        keys=np.floor((V-mn)/cell).astype(np.int64)
        kid=keys[:,0]*1_000_000+keys[:,1]*1000+keys[:,2]
        uniq,inv=np.unique(kid,return_inverse=True)
        if len(uniq)>=tgt_v: break
    # representative vertex per cell = mean of members
    newV=np.zeros((len(uniq),3))
    cnt=np.zeros(len(uniq))
    np.add.at(newV,inv,V); np.add.at(cnt,inv,1); newV/=cnt[:,None]
    newF=inv[F]
    # drop degenerate (repeated vertex) faces
    good=(newF[:,0]!=newF[:,1])&(newF[:,1]!=newF[:,2])&(newF[:,0]!=newF[:,2])
    newF=newF[good]
    # drop duplicate faces
    sortF=np.sort(newF,axis=1)
    _,uidx=np.unique(sortF,axis=0,return_index=True)
    newF=newF[np.sort(uidx)]
    return newV,newF

# ----------------------------------------------------------------- WS core (verified)
def _prep(V,F):
    V=np.asarray(V,float);F=np.asarray(F,int)
    p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    ctr=V.mean(0)
    fn=np.cross(p1-p0,p2-p0);fnu=fn/np.linalg.norm(fn,axis=1)[:,None]
    sgn=np.sign(np.einsum('ij,ij->i',fnu,(p0+p1+p2)/3-ctr));sgn[sgn==0]=1
    fnu*=sgn[:,None]
    ef=defaultdict(list)
    for fi in range(len(F)):
        vs=F[fi]
        for a,b in ((vs[0],vs[1]),(vs[1],vs[2]),(vs[2],vs[0])):
            ef[tuple(sorted((a,b)))].append(fi)
    Va,Vb,Edy=[],[],[]
    for (a,b),faces in ef.items():
        A,B=V[a],V[b];D=np.zeros((3,3))
        for fi in faces:
            nf=fnu[fi];ed=(B-A)/np.linalg.norm(B-A);ne=np.cross(ed,nf)
            third=[v for v in F[fi] if v not in (a,b)][0]
            if np.dot(ne,V[third]-(A+B)/2)>0:ne=-ne
            D+=np.outer(nf,ne)
        Va.append(A);Vb.append(B);Edy.append(D)
    return (V,F,fnu,np.array(Va),np.array(Vb),np.array(Edy))

def _pa(prep,pts,G=6.674e-11,rho=1.0,progress=False):
    V,F,fnu,Va,Vb,Edy=prep
    pts=np.asarray(pts,float);nP=len(pts)
    U=np.zeros(nP);g=np.zeros((nP,3))
    p0=V[F[:,0]];p1=V[F[:,1]];p2=V[F[:,2]]
    for m in range(nP):
        if progress and m%1000==0: print(f"    ...{m}/{nP}",flush=True)
        r=pts[m]
        Ra=Va-r;Rb=Vb-r
        ra=np.linalg.norm(Ra,axis=1);rb=np.linalg.norm(Rb,axis=1)
        el=np.linalg.norm(Vb-Va,axis=1)
        Le=np.log(np.clip((ra+rb+el)/np.clip(ra+rb-el,1e-300,None),1e-300,None))
        Er=np.einsum('kij,kj->ki',Edy,Ra)
        Ue=np.einsum('ki,ki->k',Ra,Er)*Le;ge=Er*Le[:,None]
        R0=p0-r;R1=p1-r;R2=p2-r
        r0=np.linalg.norm(R0,axis=1);r1=np.linalg.norm(R1,axis=1);r2=np.linalg.norm(R2,axis=1)
        num=np.einsum('ij,ij->i',R0,np.cross(R1,R2))
        den=r0*r1*r2+r2*np.einsum('ij,ij->i',R0,R1)+r0*np.einsum('ij,ij->i',R1,R2)+r1*np.einsum('ij,ij->i',R2,R0)
        w=2.0*np.arctan2(num,den)
        nr=np.einsum('ij,ij->i',fnu,R0);Fr=fnu*nr[:,None]
        Uf=nr*nr*w;gf=Fr*w[:,None]
        U[m]=0.5*G*rho*(Ue.sum()-Uf.sum());g[m]=G*rho*(ge.sum(0)-gf.sum(0))
    return U,g

def surface_fields(V,F,rho_gcc=None,period_hours=None,G=6.674e-20,progress=False):
    """
    Compute per-facet slope, geopotential, latitude.
    rho_gcc: bulk density in g/cm^3 (first-principles; gravity = G*rho*shape).
    Lengths MUST be in km. G defaults to km-units (6.674e-20 km^3 kg^-1 s^-2).
    Unit-safe: identical convention as density_sweep.py (verified).
    """
    N=onormals(V,F);C=centroids(V,F);prep=_prep(V,F)
    # raw WS accel with rho=1 kg/km^3, G in km-units, lengths km
    U_raw,g_raw=_pa(prep,C,G=G,rho=1.0,progress=progress)
    if rho_gcc is not None:
        rho_kg_km3 = rho_gcc*1000.0*1e9        # g/cm^3 -> kg/km^3
        U=U_raw*rho_kg_km3; g=g_raw*rho_kg_km3
    else:
        U=U_raw; g=g_raw
    if period_hours:
        om=2*math.pi/(period_hours*3600.0)
        rp=C.copy();rp[:,2]=0.0
        a_cf=-om**2*rp   # centrifugal opposes stored-outward g (verified vs analytic rotating sphere)
        pot_cf=-0.5*om**2*(C[:,0]**2+C[:,1]**2)
    else:
        a_cf=np.zeros_like(C);pot_cf=np.zeros(len(C))
    g_eff=g+a_cf
    gmag=np.linalg.norm(g_eff,axis=1)
    cos=np.einsum('ij,ij->i',g_eff,N)/np.clip(gmag,1e-300,None)
    slope=np.degrees(np.arccos(np.clip(cos,-1,1)))
    geopot=-U+pot_cf   # U stored positive-convention -> physical U = -U
    lat=np.degrees(np.arcsin(np.clip(C[:,2]/np.linalg.norm(C,axis=1),-1,1)))
    return slope,geopot,lat,gmag

# ----------------------------------------------------------------- self test
def icosphere(subdiv=3,R=1.0):
    t=(1+5**0.5)/2
    v=[[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]]
    v=[list(np.array(x)/np.linalg.norm(x)) for x in v]
    f=[[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    for _ in range(subdiv):
        mid={};nf=[]
        def mp(i,j):
            k=tuple(sorted((i,j)))
            if k in mid:return mid[k]
            mm=(np.array(v[i])+np.array(v[j]))/2;mm=mm/np.linalg.norm(mm)
            v.append(list(mm));mid[k]=len(v)-1;return mid[k]
        for a,b,c in f:
            ab,bc,ca=mp(a,b),mp(b,c),mp(c,a)
            nf+=[[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        f=nf
    return np.array(v)*R,np.array(f,int)

def ellipsoid(V,a,b,c):
    o=V.copy();o[:,0]*=a;o[:,1]*=b;o[:,2]*=c;return o

def selftest():
    print("=== SELF-TEST 1: sphere (want slope~0, |g| const) ===")
    V,F=icosphere(3,1.0)
    s,gp,lat,gm=surface_fields(V,F,rho_gcc=1.0)
    print(f"  facets {len(F)}  slope mean/max {s.mean():.3f}/{s.max():.3f} deg")
    print(f"  |g| mean/std {gm.mean():.4e}/{gm.std():.2e}   RESULT {'PASS' if s.max()<2 and gm.std()/gm.mean()<0.01 else 'CHECK'}")
    print("\n=== SELF-TEST 2: oblate a=b=1,c=0.6, no spin ===")
    V0,F0=icosphere(3,1.0);Ve=ellipsoid(V0,1,1,0.6)
    s,gp,lat,gm=surface_fields(Ve,F0,rho_gcc=1.0)
    pol=np.abs(lat)>60;eq=np.abs(lat)<30
    print(f"  slope polar {s[pol].mean():.2f}  equat {s[eq].mean():.2f} deg  (equator>polar expected)")

def report(V,F,rho_gcc,P):
    s,gp,lat,gm=surface_fields(V,F,rho_gcc=rho_gcc,period_hours=P,progress=True)
    print("\n=== SURFACE SLOPE ===")
    print(f"  mean {s.mean():.2f}  median {np.median(s):.2f}  max {s.max():.2f} deg")
    for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,30),(30,90)]:
        print(f"    {lo:2d}-{hi:2d} deg : {np.mean((s>=lo)&(s<hi))*100:5.1f}%")
    print("\n=== LATITUDINAL PROFILE (fine bins; mean & median per |lat| band) ===")
    print(f"    {'|lat|':>9} {'mean':>7} {'median':>7} {'n':>7}")
    fine=[(0,5),(5,10),(10,15),(15,20),(20,30),(30,40),(40,50),(50,60),(60,75),(75,90)]
    for lo,hi in fine:
        b=(np.abs(lat)>=lo)&(np.abs(lat)<hi)
        if b.any():
            print(f"    {lo:3d}-{hi:3d}  {s[b].mean():7.2f} {np.median(s[b]):7.2f} {b.sum():7d}")
    print("\n  (equatorial-ridge CREST = |lat|0-5 should be LOWEST;")
    print("   slope should RISE toward mid-lat then fall at poles, per Scheeres 2019)")
    print("\n=== LATITUDINAL TREND (coarse, median = ridge-robust) ===")
    for lo,hi in [(0,20),(20,40),(40,60),(60,90)]:
        b=(np.abs(lat)>=lo)&(np.abs(lat)<hi)
        if b.any(): print(f"    |lat| {lo:2d}-{hi:2d} : mean {s[b].mean():5.2f}  median {np.median(s[b]):5.2f} deg (n={b.sum()})")
    print("\n=== GEOPOTENTIAL (equator should be LOW = regolith sink) ===")
    eq=np.abs(lat)<20; pol=np.abs(lat)>60
    print(f"    equator mean : {gp[eq].mean():.4e}")
    print(f"    polar   mean : {gp[pol].mean():.4e}")

if __name__=="__main__":
    args=[a for a in sys.argv[1:] if not a.startswith('--')]
    target=None
    if '--target' in sys.argv:
        target=int(sys.argv[sys.argv.index('--target')+1])
    if len(args)>=2:
        obj=args[0];rho_gcc=float(args[1]);P=float(args[2]) if len(args)>2 else None
        print(f"Loading {obj} ...")
        V,F=load_obj(obj)
        print(f"  loaded: vertices {len(V)}  facets {len(F)}")
        if target is None: target=8000
        if len(F)>target:
            print(f"  decimating {len(F)} -> ~{target} facets for pilot ...")
            V,F=decimate_cluster(V,F,target)
            print(f"  after decimation: vertices {len(V)}  facets {len(F)}")
        print(f"  rho={rho_gcc} g/cm^3  period={P}h  computing (this may take a few min)...")
        report(V,F,rho_gcc,P)
    else:
        selftest()
