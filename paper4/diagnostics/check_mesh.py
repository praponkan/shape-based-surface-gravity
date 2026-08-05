#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mesh quality check — detect degenerate/flipped facets that produce fake slopes.
Especially important for bilobed shapes (Itokawa) after decimation.
USAGE: python check_mesh.py <obj> [--target N]
"""
import sys, numpy as np
from surface_slope import load_obj, decimate_cluster, onormals, centroids

obj=sys.argv[1]
target=None
if '--target' in sys.argv: target=int(sys.argv[sys.argv.index('--target')+1])
V,F=load_obj(obj)
print(f"loaded: vertices {len(V)} facets {len(F)}")
if target and len(F)>target:
    V,F=decimate_cluster(V,F,target)
    print(f"decimated -> facets {len(F)}")

p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
# facet areas
cr=np.cross(p1-p0,p2-p0)
area=0.5*np.linalg.norm(cr,axis=1)
print(f"\nfacet area: min {area.min():.3e} max {area.max():.3e} median {np.median(area):.3e} km^2")
print(f"  degenerate (area<1e-10): {(area<1e-10).sum()}")
# aspect ratio (longest edge / shortest) - sliver detection
e0=np.linalg.norm(p1-p0,axis=1);e1=np.linalg.norm(p2-p1,axis=1);e2=np.linalg.norm(p0-p2,axis=1)
edges=np.stack([e0,e1,e2],axis=1)
aspect=edges.max(1)/np.clip(edges.min(1),1e-30,None)
print(f"aspect ratio: median {np.median(aspect):.1f} max {aspect.max():.1f}")
print(f"  slivers (aspect>20): {(aspect>20).sum()} ({(aspect>20).mean()*100:.1f}%)")
# normal consistency: do normals point outward? check vs centroid
N=onormals(V,F); C=centroids(V,F); ctr=V.mean(0)
outward=np.einsum('ij,ij->i',N,C-ctr)
print(f"normals pointing inward (possible flip): {(outward<0).sum()} ({(outward<0).mean()*100:.1f}%)")
# for a star-convex-ish body most should be outward; bilobed neck may have some
print("")
print("")

# --- watertight / manifold check (
def watertight_check(V,F):
    from collections import defaultdict
    edge_count=defaultdict(int)
    for f in F:
        for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):
            edge_count[tuple(sorted((a,b)))]+=1
    counts=np.array(list(edge_count.values()))
    boundary=(counts==1).sum()      #   ( )
    manifold=(counts==2).sum()      #  
    nonmanifold=(counts>2).sum()    #  
    print(f"\n=== WATERTIGHT / MANIFOLD CHECK ===")
    print(f"  edges total     : {len(counts)}")
    print(f"  manifold (=2)   : {manifold} ({manifold/len(counts)*100:.1f}%)")
    print("")
    print("")
    # duplicate vertices
    #
    uniq=len(set(map(tuple,np.round(V,9))))
    print(f"  vertices        : {len(V)} (unique coords: {uniq}, dup: {len(V)-uniq})")
    if boundary==0 and nonmanifold==0:
        print("")
    else:
        print("")
        print("")

watertight_check(V,F)
