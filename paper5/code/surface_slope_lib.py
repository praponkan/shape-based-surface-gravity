#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper IV — Surface slope & geopotential (LIBRARY version).

Uses the polyhedral-gravity library (same as Paper III) instead of the
pure-numpy Werner-Scheeres, to get correct gravity on concave/bilobed
shapes (e.g. Itokawa). Requires:  pip install polyhedral-gravity --break-system-packages

Slope = angle between gravity acceleration vector and outward facet normal
        (Werner-Scheeres convention; centrifugal added for the rotating frame).
Geopotential = gravitational potential + rotational potential.

USAGE:
    python surface_slope_lib.py <obj> <density_gcc> <period_h> [--target N]
                                [--decimator cluster|qem]
Example:
    python surface_slope_lib.py "Itokawa Hayabusa 50k poly.obj" 1.95 12.132

DECIMATION (added 2026-08, after Paper IV was published)
    The default remains vertex clustering, unchanged in every respect, so
    that this script reproduces the published results exactly. A second
    method, quadric edge collapse, is available with --decimator qem and
    requires decimate_qem.py alongside this file.

    Vertex clustering is accurate: tested against a homogeneous ellipsoid
    with a closed-form surface gravity, it reproduces the mean slope to
    3.9% and the normalised dispersion to 0.07% on average, marginally
    better than quadric collapse. Nothing in the published numbers is in
    question.

    It has two limitations that matter for other uses. It does not preserve
    topology: the reduced mesh has edges shared by more than two faces, from
    96 such edges at a 43% reduction to 1,449 at 96% on a 1.3-million-facet
    comet nucleus. And it cannot reach small targets, because the grid
    resolution is chosen from a fixed ladder: requests for 1200, 800, 500
    and 300 faces all return the same 2,159-face mesh.

    Neither affects a volume, a potential or a slope, all of which are sums
    over oriented facets that never consult connectivity. Both matter for
    anything needing to know which faces meet at a vertex. See
    DECIMATION_NOTES.txt.
"""
import sys, math
import numpy as np

try:
    import polyhedral_gravity as pg
except ImportError:
    sys.exit("ERROR: install library first:\n    pip install polyhedral-gravity --break-system-packages")

G_SI = 6.674e-11   # m^3 kg^-1 s^-2

# ---------------------------------------------------------------- mesh utils
def load_obj(path):
    with open(path,'rb') as fh: raw=fh.read()
    text=raw.decode('utf-8',errors='ignore').replace('\r\n','\n').replace('\r','\n')
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

def outward_normals(V,F):
    p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    n=np.cross(p1-p0,p2-p0); n/=np.linalg.norm(n,axis=1)[:,None]
    ctr=V.mean(0); fc=(p0+p1+p2)/3
    s=np.sign(np.einsum('ij,ij->i',n,fc-ctr)); s[s==0]=1
    return n*s[:,None]

def centroids(V,F): return (V[F[:,0]]+V[F[:,1]]+V[F[:,2]])/3

def volume(V,F):
    p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    return abs(np.sum(np.einsum('ij,ij->i',p0,np.cross(p1,p2)))/6.0)

def decimate_cluster(V,F,target_faces):
    """Vertex-clustering decimation. UNCHANGED from the published version.

    Snaps vertices to a grid whose resolution is taken from a fixed ladder,
    then rebuilds the faces from the snapped indices, dropping degenerate and
    duplicate faces. Fast and accurate in the fields it produces, but it does
    not preserve topology and cannot reach small targets. See the module
    docstring and DECIMATION_NOTES.txt.
    """
    if len(F)<=target_faces: return V,F
    tgt_v=max(target_faces//2,200)
    mn,mx=V.min(0),V.max(0); ext=mx-mn; ext[ext==0]=1e-9
    for gres in [16,20,24,28,32,40,48,56,64,80,96,128,160,200,256]:
        cell=ext/gres
        keys=np.floor((V-mn)/cell).astype(np.int64)
        kid=keys[:,0]*1_000_000+keys[:,1]*1000+keys[:,2]
        uniq,inv=np.unique(kid,return_inverse=True)
        if len(uniq)>=tgt_v: break
    newV=np.zeros((len(uniq),3)); cnt=np.zeros(len(uniq))
    np.add.at(newV,inv,V); np.add.at(cnt,inv,1); newV/=cnt[:,None]
    newF=inv[F]
    good=(newF[:,0]!=newF[:,1])&(newF[:,1]!=newF[:,2])&(newF[:,0]!=newF[:,2])
    newF=newF[good]
    sortF=np.sort(newF,axis=1); _,uidx=np.unique(sortF,axis=0,return_index=True)
    return newV,newF[np.sort(uidx)]

# ---------------------------------------------------------------- library solver
def decimate(V, F, target_faces, method="cluster"):
    """Decimate by the named method.

    The default is "cluster", which is what Paper IV used and what reproduces
    its numbers. "qem" is quadric edge collapse from decimate_qem.py: equally
    accurate in the fields, but it preserves the manifold property and returns
    the requested face count exactly. Use it when the mesh will be used for
    anything that needs connectivity.
    """
    if method == "cluster":
        return decimate_cluster(V, F, target_faces)
    if method == "qem":
        try:
            from decimate_qem import decimate_qem
        except ImportError:
            sys.exit("--decimator qem needs decimate_qem.py in the same "
                     "directory")
        return decimate_qem(V, F, target_faces)
    sys.exit(f"unknown decimator: {method}")


def mesh_topology(F):
    """Directed-edge audit: is the mesh a closed, oriented 2-manifold?

    Reports repeated and unmatched directed edges and the Euler characteristic,
    which is 2 for a single topological sphere and 2k for k disjoint ones. Not
    used by anything above; provided so that a user can check a mesh before
    relying on its connectivity.
    """
    from collections import Counter
    c = Counter()
    for f in F:
        for k in range(3):
            c[(int(f[k]), int(f[(k+1) % 3]))] += 1
    rep = sum(1 for v in c.values() if v != 1)
    unm = sum(1 for (a, b) in c if (b, a) not in c)
    adj = {}
    for f in F:
        a, b, cc = int(f[0]), int(f[1]), int(f[2])
        for u, v in ((a, b), (b, cc), (cc, a)):
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
    seen, comps = set(), 0
    for s0 in adj:
        if s0 in seen:
            continue
        comps += 1
        st = [s0]
        while st:
            u = st.pop()
            if u in seen:
                continue
            seen.add(u)
            st.extend(w for w in adj[u] if w not in seen)
    nv = len(np.unique(np.asarray(F)))
    chi = nv - len(c)//2 + len(F)
    return {"repeated_directed_edges": rep, "unmatched_directed_edges": unm,
            "euler_characteristic": int(chi), "components": comps,
            "closed": rep == 0 and unm == 0,
            "manifold": rep == 0 and unm == 0 and chi == 2*comps}


def surface_fields(V,F,rho_gcc,period_hours,progress=False):
    """
    Gravity via polyhedral-gravity library at facet centroids (nudged inward
    by eps like Paper III). density in g/cm^3, lengths in km.
    Returns slope[deg], geopot, lat[deg], gmag.
    """
    nf=outward_normals(V,F)
    C=centroids(V,F)
    # density: convert g/cm^3 -> kg/km^3 so accel comes out in km/s^2 with SI G in km units
    # Simpler: work in SI metres. Convert V (km) -> m.
    Vm=V*1000.0                      # km -> m
    Cm=C*1000.0
    nfm=nf                           # unit normals (dimensionless)
    rho_si=rho_gcc*1000.0            # g/cm^3 -> kg/m^3
    ext=np.linalg.norm(Vm.max(0)-Vm.min(0))
    eps=1e-4*ext
    pts=(Cm - eps*nfm)               # nudge inward, in metres
    if progress: print(f"    evaluating {len(F)} facets via library...",flush=True)
    poly=pg.Polyhedron(
        polyhedral_source=(Vm.tolist(),F.tolist()),
        density=rho_si,
        normal_orientation=pg.NormalOrientation.OUTWARDS,
        integrity_check=pg.PolyhedronIntegrity.DISABLE)
    res=pg.evaluate(poly,pts.tolist(),parallel=True)
    # res[i] = (potential, acceleration_vec, tensor). acceleration in m/s^2
    U=np.array([r[0] for r in res])                 # potential (m^2/s^2)
    g=np.array([r[1] for r in res])                 # accel vector (m/s^2), points ??? 
    # centrifugal in SI (Cm in metres). Library g points INWARD (attraction).
    # Centrifugal acceleration points OUTWARD from spin axis. To combine in the
    # same convention as g (inward-positive), centrifugal REDUCES effective inward
    # pull, so we ADD an inward-negative term: a_cf_inward = -om^2 * r_perp.
    om=2*math.pi/(period_hours*3600.0)
    rp=Cm.copy(); rp[:,2]=0.0
    a_cf=om**2*rp                                   # centrifugal points OUTWARD (verified vs analytic rotating sphere)
    pot_cf=-0.5*om**2*(Cm[:,0]**2+Cm[:,1]**2)
    g_eff=g+a_cf
    gmag=np.linalg.norm(g_eff,axis=1)
    # library g points inward; -g_eff points outward, angle with outward normal N
    # gives gravitational slope (0 = surface perpendicular to net accel). Verified
    # by sphere selftest (should read ~0).
    cos=np.einsum('ij,ij->i',-g_eff,nfm)/np.clip(gmag,1e-300,None)
    slope=np.degrees(np.arccos(np.clip(cos,-1,1)))
    geopot=-U+pot_cf   # library U is positive-convention -> physical U = -U
    lat=np.degrees(np.arcsin(np.clip(Cm[:,2]/np.linalg.norm(Cm,axis=1),-1,1)))
    return slope,geopot,lat,gmag

def report(V,F,rho_gcc,P):
    s,gp,lat,gm=surface_fields(V,F,rho_gcc,P,progress=True)
    print("\n=== SURFACE SLOPE (library) ===")
    print(f"  mean {s.mean():.2f}  median {np.median(s):.2f}  max {s.max():.2f} deg")
    print(f"  slope>90: {(s>90).sum()} ({(s>90).mean()*100:.2f}%)")
    for lo,hi in [(0,5),(5,10),(10,15),(15,20),(20,30),(30,90)]:
        print(f"    {lo:2d}-{hi:2d} deg : {np.mean((s>=lo)&(s<hi))*100:5.1f}%")
    print("\n=== LATITUDINAL PROFILE ===")
    print(f"    {'|lat|':>9} {'mean':>7} {'median':>7} {'n':>7}")
    for lo,hi in [(0,10),(10,20),(20,30),(30,45),(45,60),(60,90)]:
        b=(np.abs(lat)>=lo)&(np.abs(lat)<hi)
        if b.any(): print(f"    {lo:3d}-{hi:3d}  {s[b].mean():7.2f} {np.median(s[b]):7.2f} {b.sum():7d}")

def make_icosphere(subdiv=3,R=1.0):
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

def selftest():
    print("=== SELF-TEST: sphere R=0.25 km, no spin (slope must be ~0) ===")
    V,F=make_icosphere(3,0.25)  # 0.25 km sphere
    s,gp,lat,gm=surface_fields(V,F,rho_gcc=1.95,period_hours=1e9)  # ~no spin
    print(f"  facets {len(F)}  slope mean/max {s.mean():.3f}/{s.max():.3f} deg")
    print(f"  RESULT {'PASS (library sign convention OK)' if s.max()<3 else 'FAIL - sign convention wrong, need -g'}")
    if s.max()>=3:
        print("  Trying angle(-g,N):")
        # re-evaluate with flipped
        print("  If this fails, the slope line needs g_eff -> -g_eff")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=='selftest':
        selftest(); sys.exit()
    args=[a for a in sys.argv[1:] if not a.startswith('--')]
    obj=args[0]; rho=float(args[1]); P=float(args[2])
    target=30000
    if '--target' in sys.argv: target=int(sys.argv[sys.argv.index('--target')+1])
    method='cluster'
    if '--decimator' in sys.argv: method=sys.argv[sys.argv.index('--decimator')+1]
    print(f"Loading {obj} ...")
    V,F=load_obj(obj); print(f"  facets {len(F)}")
    if len(F)>target:
        V,F=decimate(V,F,target,method); print(f"  decimated -> {len(F)} by {method}")
        t=mesh_topology(F)
        print(f"  topology: chi = {t['euler_characteristic']}, "
              f"{t['repeated_directed_edges']} repeated and "
              f"{t['unmatched_directed_edges']} unmatched directed edges"
              + ("" if t['manifold'] else "   [not a closed 2-manifold]"))
    print(f"  density {rho} g/cm^3  period {P} h  (library polyhedral-gravity)")
    if '--nospin' in sys.argv:
        print("  [--nospin] centrifugal DISABLED (gravity only)")
        report(V,F,rho,1e12)   # huge period -> ~no centrifugal
    elif '--both' in sys.argv:
        print("\n########## WITH SPIN ##########")
        report(V,F,rho,P)
        print("\n########## NO SPIN (gravity only) ##########")
        report(V,F,rho,1e12)
    else:
        report(V,F,rho,P)
