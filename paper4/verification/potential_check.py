#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POTENTIAL SANITY CHECK for Paper V.
Verify the polyhedral-library gravitational potential against the exact analytic
potential of a uniform-density ellipsoid (MacMillan/Kellogg). If they match,
our potential (hence potential variance used in Paper V) is correct in an
ABSOLUTE sense -- independent of JAXA.

USAGE: python potential_check.py
Needs: polyhedral-gravity, scipy, numpy.
"""
import numpy as np, math
from scipy import integrate
try:
    import polyhedral_gravity as pg
except ImportError:
    raise SystemExit("need polyhedral-gravity library")

G=6.674e-11  # SI

# ---------- analytic ellipsoid potential ----------
def ellipsoid_potential(x,y,z,a,b,c,rho):
    def integrand_t(t):
        u=t*t
        Delta=np.sqrt((a*a+u)*(b*b+u)*(c*c+u))
        return (1 - x*x/(a*a+u) - y*y/(b*b+u) - z*z/(c*c+u))/Delta*2*t
    val,_=integrate.quad(integrand_t,0,np.inf,limit=400)
    return -np.pi*G*rho*a*b*c*val

# ---------- triangulated ellipsoid mesh ----------
def icosphere(subdiv,R=1.0):
    t=(1+5**0.5)/2
    v=[[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]]
    v=[list(np.array(x)/np.linalg.norm(x)) for x in v]
    f=[[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]]
    for _ in range(subdiv):
        mid={};nf=[]
        def mp(i,j):
            k=tuple(sorted((i,j)))
            if k in mid:return mid[k]
            m=(np.array(v[i])+np.array(v[j]))/2;m/=np.linalg.norm(m)
            v.append(list(m));mid[k]=len(v)-1;return mid[k]
        for a,b,c in f:
            ab,bc,ca=mp(a,b),mp(b,c),mp(c,a)
            nf+=[[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        f=nf
    return np.array(v)*R,np.array(f,int)

def make_ellipsoid(sub,a,b,c):
    V,F=icosphere(sub,1.0)
    V=V*np.array([a,b,c])
    return V,F

# ---------- run comparison ----------
a,b,c = 1500.0,1000.0,800.0   # metres
rho = 2000.0                   # kg/m^3
V,F = make_ellipsoid(4,a,b,c)  # metres
print(f"ellipsoid a,b,c={a},{b},{c} m, rho={rho}, facets={len(F)}")

# library potential at facet centroids, nudged slightly inward
p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
C=(p0+p1+p2)/3
n=np.cross(p1-p0,p2-p0); n/=np.linalg.norm(n,axis=1)[:,None]
ctr=V.mean(0); fc=C
s=np.sign(np.einsum('ij,ij->i',n,fc-ctr)); s[s==0]=1; n=n*s[:,None]
ext=np.linalg.norm(V.max(0)-V.min(0)); eps=1e-4*ext
pts=C - eps*n

poly=pg.Polyhedron(polyhedral_source=(V.tolist(),F.tolist()),density=rho,
    normal_orientation=pg.NormalOrientation.OUTWARDS,
    integrity_check=pg.PolyhedronIntegrity.DISABLE)
res=pg.evaluate(poly,pts.tolist(),parallel=True)
U_lib=np.array([r[0] for r in res])

# analytic at same points (on surface, eps negligible)
U_ana=np.array([ellipsoid_potential(x,y,z,a,b,c,rho) for (x,y,z) in C])

# library sign convention may be +; align by matching sphere sign (both should be negative wells)
# compare both raw and sign-flipped
for label,Ul in [("library U",U_lib),("-library U",-U_lib)]:
    diff=Ul-U_ana
    r=np.corrcoef(Ul,U_ana)[0,1]
    rel=np.abs(diff)/np.abs(U_ana)
    print(f"\n{label} vs analytic:")
    print(f"  Pearson r = {r:.6f}")
    print(f"  mean rel.err = {rel.mean()*100:.3f}%   max rel.err = {rel.max()*100:.3f}%")

print("\n=> if |r|~1 and rel.err<1%, our potential is CORRECT in absolute terms")
print("   (Paper V variance rests on a verified potential).")
