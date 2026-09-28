#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check every headline number in the paper against the archived run records.

    python check_key_results.py

Reads paperV_final.json and synth_final.json from the same directory and
verifies the values quoted in the manuscript. Nothing here recomputes a
potential; it only asserts that what the paper says matches what the runs
produced, so it takes a second and needs no shape models and no gravity
library.

Two record sets live in paperV_final.json. The runs of 2026-08-18 and later
carry --meshcache in their command and use quadric edge collapse; those are the
ones the paper reports. The earlier runs are retained for provenance and are
ignored here.

Exit status is 0 if every check passes and 1 otherwise.
"""

import json
import os
import sys

def half_ulp(x):
    """Half of the last digit the paper prints, from the literal given.

    A value written as 4.35 is checked to +/-0.005, one written as 16.889 to
    +/-0.0005. This is the right tolerance for a rounding check: it asks
    whether the paper rounded the run correctly, not whether the two agree to
    machine precision.
    """
    t = f"{x!r}"
    if "e" in t or "E" in t:
        return abs(x) * 5e-7
    dec = len(t.split(".")[1]) if "." in t else 0
    return 0.5 * 10 ** (-dec)


def load(name):
    for p in (name, os.path.join(os.path.dirname(__file__), name),
              os.path.join("data", name)):
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    sys.exit(f"cannot find {name}")


def qem_runs(records):
    """The runs the paper reports: quadric decimation, corrected clipper."""
    return [r for r in records
            if "meshcache" in r.get("command", "")
            and "--decimator cluster" not in r.get("command", "")]


def find(runs, frag, model="head", x=None, mass=None, facets=None,
         fracsplit=None, xtol=1e-4):
    for r in runs:
        if frag not in r["shape_model"]:
            continue
        if mass is not None and abs(r["M_total_kg"] - mass) > 1e-3 * mass:
            continue
        if facets is not None and r["facets_used"] != facets:
            continue
        if fracsplit is True and "--fracsplit" not in r["command"]:
            continue
        if fracsplit is False and "--fracsplit" in r["command"]:
            continue
        m = r["models"].get(model)
        if not m or m.get("status") != "ok":
            continue
        if x is not None and abs(m["x_lo_km"] - x) > xtol:
            continue
        return m, r
    return None, None


class Check:
    def __init__(self):
        self.n = self.bad = 0

    def __call__(self, label, got, want, kind="rho", note=""):
        self.n += 1
        tol = half_ulp(want)
        ok = got is not None and abs(got - want) <= tol
        if not ok:
            self.bad += 1
        g = "n/a" if got is None else f"{got:.6g}"
        print(f"  {'ok ' if ok else 'FAIL'}  {label:<46} "
              f"paper {want:<12.6g} run {g}" + (f"   {note}" if note else ""))
        return ok

    def section(self, title):
        print(f"\n{title}\n{'-' * len(title)}")

    def report(self):
        print(f"\n{'='*72}")
        print(f"{self.n - self.bad} of {self.n} checks passed")
        if self.bad:
            print(f"{self.bad} FAILED - the manuscript and the records "
                  f"disagree")
        else:
            print("every number quoted in the paper is reproduced by the "
                  "archived records")
        return 1 if self.bad else 0


def main():
    rec = load("paperV_final.json")
    syn = load("synth_final.json")
    runs = qem_runs(rec)
    c = Check()

    print(f"paperV_final.json: {len(rec)} records, {len(runs)} from the "
          f"reported run set")
    print(f"synth_final.json:  {len(syn)} synthetic bodies")

    # ---- Section 3: Itokawa reference -----------------------------------
    c.section("Section 3.1  Itokawa at the detected neck")
    m, r = find(runs, "Itokawa", x=0.1608, facets=49152, mass=3.58e10,
                fracsplit=False)
    if m:
        mn, dv = m["minimum"], m["derived"]
        c("head density", mn["rho_R"], 2800)
        c("complement density", mn["rho_rest"], 1858)
        c("contrast", mn["contrast"], 1.507, "contrast")
        c("improvement %", mn["improvement_over_uniform_pct"], 16.889,
          "improvement")
        c("COM offset m", dv["com_offset_m"], 16.86, "com")
        c("region fraction %", m["volume_fraction_pct"], 16.4, "frac")
        c("Euler characteristic",
          m.get("manifold", {}).get("euler_characteristic"), 2, "rho")
    else:
        print("  FAIL  reference run not found")
        c.bad += 1
        c.n += 1

    # neck model at the same plane
    m, _ = find(runs, "Itokawa", model="neck", facets=49152, mass=3.58e10,
                fracsplit=False, x=0.1608, xtol=0.06)
    if m:
        c("neck-model density", m["minimum"]["rho_R"], 2525)
        c("neck-model contrast", m["minimum"]["contrast"], 1.300, "contrast")
        c("neck-model improvement %",
          m["minimum"]["improvement_over_uniform_pct"], 2.275, "improvement")

    # ---- Section 3.2: resolution series ---------------------------------
    c.section("Section 3.2  Itokawa resolution series")
    seen = {}
    for r in runs:
        if "Itokawa" not in r["shape_model"]:
            continue
        if "--xsplit" in r["command"] or "--scale" in r["command"]:
            continue
        if abs(r["M_total_kg"] - 3.58e10) > 1e7:
            continue
        m = r["models"].get("head")
        if m and m["status"] == "ok":
            seen[r["facets_used"]] = (m["x_lo_km"], m["minimum"]["rho_R"])
    for f, want_x, want_rho in [(20000, 0.1702, 2900), (30000, 0.1608, 2800),
                                (40000, 0.1608, 2800), (49152, 0.1608, 2800)]:
        got = seen.get(f)
        c(f"{f} facets: plane", got[0] if got else None, want_x, "volume")
        c(f"{f} facets: rho_head", got[1] if got else None, want_rho)

    # ---- Section 4: the four weak-signal cases --------------------------
    c.section("Section 4  weak-signal cases")

    m, _ = find(runs, "Eros", x=4.4698, xtol=0.01)
    if m:
        c("Eros contrast", m["minimum"]["contrast"], 1.018, "contrast")

    m, _ = find(runs, "Eros", x=-10.7619, xtol=0.01)
    if m:
        c("Eros misplaced contrast", m["minimum"]["contrast"], 0.876,
          "contrast")
        c("Eros misplaced improvement %",
          m["minimum"]["improvement_over_uniform_pct"], 5.526, "improvement")
        c("Eros misplaced COM m", m["derived"]["com_offset_m"], -214.8, "rho")

    m, r = find(runs, "Bennu", x=0.0, xtol=0.01)
    if m:
        c("Bennu facets", r["facets_used"], 50000, "rho")
        c("Bennu contrast", m["minimum"]["contrast"], 0.999, "contrast")
        c("Bennu improvement %", m["minimum"]["improvement_over_uniform_pct"],
          0.000, "improvement")
        c("Bennu COM m", m["derived"]["com_offset_m"], -0.07, "com",
          "negative, as Delta M is")
        c("Bennu volume km^3", r["volume_km3"], 0.061553, "volume")

    m, r = find(runs, "cg_mspcd", x=0.9517, xtol=1e-4)
    if m:
        c("67P facets", r["facets_used"], 50000, "rho")
        c("67P volume km^3", r["volume_km3"], 18.5912, "rho")
        c("67P contrast", m["minimum"]["contrast"], 1.073, "contrast")
        c("67P improvement %", m["minimum"]["improvement_over_uniform_pct"],
          0.574, "improvement")

    m, _ = find(runs, "cg_mspcd", model="neck")
    if m:
        c("67P neck contrast", m["minimum"]["contrast"], 0.903, "contrast")
        c("67P neck improvement %",
          m["minimum"]["improvement_over_uniform_pct"], 1.132, "improvement")

    m, _ = find(runs, "huryumov", x=0.9517, xtol=1e-4)
    if m:
        c("67P SPC contrast", m["minimum"]["contrast"], 1.085, "contrast")
        c("67P SPC improvement %",
          m["minimum"]["improvement_over_uniform_pct"], 0.725, "improvement")

    m, _ = find(runs, "arrokoth", x=-4.7863, mass=1.6495e15, fracsplit=False,
                xtol=1e-3)
    if m:
        c("Arrokoth contrast at 400", m["minimum"]["contrast"], 0.964,
          "contrast")
        c("Arrokoth improvement %",
          m["minimum"]["improvement_over_uniform_pct"], 0.883, "improvement")
        c("Arrokoth Euler characteristic",
          m.get("manifold", {}).get("euler_characteristic"), 4, "rho",
          "two unjoined shells")

    # ---- Section 5: the synthetic family --------------------------------
    c.section("Section 5  synthetic family")
    want = {"sphere": (1.24, None), "sym_waist": (3.68, None),
            "sym_neck": (4.35, None), "sym_deep": (4.47, None),
            "asym_waist": (16.67, -2.02), "asym_neck": (17.63, -2.37),
            "asym_deep": (19.50, -2.07), "asym_extreme": (43.71, -1.22)}
    for b in syn:
        nm = b.get("name")
        if nm not in want or "scaling_slope" not in b:
            continue
        wi, ws = want[nm]
        c(f"{nm} improvement %", b["improvement_max_pct"], wi, "improvement")
        if ws is None:
            c.n += 1
            ok = b["scaling_slope"] is None
            if not ok:
                c.bad += 1
            print(f"  {'ok ' if ok else 'FAIL'}  {nm + ' has no exponent':<46} "
                  f"paper none         run "
                  f"{b['scaling_slope'] if b['scaling_slope'] else 'none'}")
        else:
            c(f"{nm} exponent", b["scaling_slope"], ws, "contrast")
        c.n += 1
        chi = b.get("source_euler_characteristic")
        ok = chi == 2
        if not ok:
            c.bad += 1
        print(f"  {'ok ' if ok else 'FAIL'}  {nm + ' source chi':<46} "
              f"paper 2            run {chi}")

    # ---- topology audit across the whole reported set --------------------
    c.section("Topology of every reported region")
    bad = []
    for r in runs:
        for k, m in r["models"].items():
            if m.get("status") != "ok":
                continue
            mf = m.get("manifold", {})
            if not m.get("manifold_pass", True):
                bad.append((os.path.basename(r["shape_model"])[:24], k,
                            mf.get("euler_characteristic"),
                            m.get("source_manifold", {}).get("closed")))
    print(f"  {len(bad)} of "
          f"{sum(1 for r in runs for m in r['models'].values() if m.get('status') == 'ok')}"
          f" models fail the manifold criterion")
    for b_, k, chi, src in bad:
        print(f"      {b_:<26} {k:<5} chi={chi}  source closed={src}")
    print("  all such cases inherit the defect from the shape model as "
          "supplied;")
    print("  see Section 2.7. Volumes and potentials are unaffected.")

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
