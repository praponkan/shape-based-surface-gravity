"""
compute_invariants.py - shape-based surface gravity, Paper III preparation
=========================================
For every shape model (.obj/.ply/.stl/.off) in a folder, compute the
deformation invariants and elongation used by the w-blend law:

    Psi = R_A / R_V            (elongation; R_A=sqrt(A/4pi), R_V=(3V/4pi)^(1/3))
    s_i = ln(axis_i / gmean)   from the inertia-equivalent ellipsoid (DEEVE)
    P2  = sum s_i^2            (deformation magnitude)
    P3  = sum s_i^3            (oblate < 0  /  prolate > 0)

Also flags whether each body is "deformation-clear" (P2 >= 0.10), which is the
regime where the optimal weight w* is well-defined (below it, w* is a 0/0 form).

Writes invariants_results.json (send this back) and prints a table.

RAM-safe: large meshes are decimated before the (cheap) area/volume/inertia
computation, so didymos/dimorphos (~225 MB) will not exhaust 4 GB.

Usage:
    python compute_invariants.py                 (scans current folder, recursive)
    python compute_invariants.py --dir shapes
Requires: numpy, trimesh (already installed).
"""
import argparse, glob, json, math, os, sys, gc
import numpy as np

try:
    import trimesh
except ImportError:
    sys.exit("trimesh is required (pip install trimesh)")

TARGET_FACES = 40000          # decimate above this (area/vol/inertia are robust)


def equivalent_ellipsoid_axes(mesh):
    """Inertia-equivalent (DEEVE) ellipsoid semi-axes a>=b>=c, volume-matched,
    for a uniform-density body. Same convention used for P2,P3 throughout this work."""
    m = mesh.volume                              # rho = 1
    I = mesh.moment_inertia
    evals = np.sort(np.linalg.eigh(I)[0])        # I_a <= I_b <= I_c
    Ia, Ib, Ic = evals
    a2 = 2.5 / m * (-Ia + Ib + Ic)
    b2 = 2.5 / m * ( Ia - Ib + Ic)
    c2 = 2.5 / m * ( Ia + Ib - Ic)
    if min(a2, b2, c2) <= 0:
        return None
    axes = np.sqrt([a2, b2, c2])
    scale = (mesh.volume / (4.0/3.0*math.pi*axes.prod())) ** (1.0/3.0)
    return np.sort(axes)[::-1] * scale           # a >= b >= c


def invariants(axes):
    ax = np.asarray(axes, float)
    g = np.prod(ax) ** (1.0/3.0)                 # geometric mean
    s = np.log(ax / g)                           # sum s = 0 by construction
    return float(np.sum(s**2)), float(np.sum(s**3))


def load_mesh(path, target=TARGET_FACES):
    mesh = trimesh.load(path, force="mesh", process=True)
    n0 = len(mesh.faces)
    if n0 > 400000:
        mesh = mesh.simplify_quadric_decimation(face_count=120000)
        mesh.process(); gc.collect()
    if len(mesh.faces) > target * 1.2:
        mesh = mesh.simplify_quadric_decimation(face_count=target)
        mesh.process()
    return mesh, n0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".", help="folder with shape models (recursive)")
    ap.add_argument("--out", default="invariants_results.json")
    ap.add_argument("--max-mb", type=float, default=400.0,
                    help="skip files larger than this many MB")
    args = ap.parse_args()

    exts = ("*.obj", "*.ply", "*.stl", "*.off", "*.OBJ", "*.PLY", "*.STL")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.dir, "**", e), recursive=True)
    files = sorted(set(files))
    if not files:
        sys.exit("no shape files found under '%s'" % os.path.abspath(args.dir))

    print("%-26s %7s %8s %8s %8s  %-6s %s" %
          ("body", "Psi", "P2", "P3", "faces", "clear", "topo_ok"))
    print("-" * 78)
    results = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        size_mb = os.path.getsize(path) / 1e6
        if size_mb > args.max_mb:
            print("%-26s  SKIP (%.0f MB > --max-mb)" % (name[:26], size_mb))
            results.append({"name": name, "file": os.path.basename(path),
                            "error": "skipped, %.0f MB" % size_mb})
            continue
        try:
            mesh, n0 = load_mesh(path)
        except Exception as e:
            print("%-26s  LOAD FAIL: %s" % (name[:26], e))
            results.append({"name": name, "file": os.path.basename(path),
                            "error": "load fail: %s" % e})
            continue

        V = float(mesh.volume)
        A = float(mesh.area)
        R_V = (3.0*V/(4.0*math.pi)) ** (1.0/3.0)
        R_A = math.sqrt(A/(4.0*math.pi))
        Psi = R_A / R_V
        axes = equivalent_ellipsoid_axes(mesh)
        rec = {"name": name, "file": os.path.basename(path),
               "faces": len(mesh.faces), "faces_orig": n0,
               "volume": V, "area": A, "R_V": R_V, "R_A": R_A, "Psi": Psi,
               "watertight": bool(mesh.is_watertight),
               "euler": int(mesh.euler_number)}
        if axes is None:
            rec["error"] = "inertia-ellipsoid solve failed (non-physical)"
            print("%-26s %7.4f  ellipsoid solve failed" % (name[:26], Psi))
        else:
            P2, P3 = invariants(axes)
            rec.update({"axes": axes.tolist(), "P2": P2, "P3": P3})
            clear = P2 >= 0.10
            topo_ok = bool(mesh.is_watertight) and int(mesh.euler_number) == 2
            rec["deformation_clear"] = clear
            print("%-26s %7.4f %8.4f %+8.4f %8d  %-6s %s"
                  % (name[:26], Psi, P2, P3, len(mesh.faces),
                     "YES" if clear else "no", "yes" if topo_ok else "NO"))
        results.append(rec)
        del mesh; gc.collect()

    usable = [r for r in results if r.get("deformation_clear")]
    print("-" * 78)
    print("total bodies: %d | deformation-clear (P2>=0.10, w* usable): %d"
          % (len([r for r in results if "P2" in r]), len(usable)))

    json.dump({"results": results}, open(args.out, "w"), indent=1)
    print("\nwrote %s  --  send this file back." % args.out)


if __name__ == "__main__":
    main()

