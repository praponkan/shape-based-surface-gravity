import numpy as np
from surface_slope import load_obj
import sys
V,F=load_obj(sys.argv[1])
ext=V.max(0)-V.min(0)
ctr=V.mean(0)
print(f"vertices {len(V)} facets {len(F)}")
print(f"extent  x={ext[0]:.4f}  y={ext[1]:.4f}  z={ext[2]:.4f} km")
print(f"center  {ctr}")
#
axis_names=['x','y','z']
short=np.argmin(ext); long=np.argmax(ext)
print(f"shortest axis: {axis_names[short]} (spin axis should be here for oblate)")
print(f"longest axis:  {axis_names[long]}")
if short==2:
    print("=> spin axis = z. GOOD, latitude computation valid.")
else:
    print(f"=> WARNING: shortest axis is {axis_names[short]}, not z!")
    print("   Bennu OBJ may use different axis convention -> latitude wrong.")
    print("   Need to rotate so spin axis -> z before slope-latitude analysis.")
# inertia tensor
Vc=V-ctr
I=np.zeros((3,3))
for p in Vc:
    I+=np.dot(p,p)*np.eye(3)-np.outer(p,p)
w,vec=np.linalg.eigh(I)
print(f"inertia eigenvalues: {w}")
print(f"max-inertia axis (true spin axis): {vec[:,np.argmax(w)]}")
