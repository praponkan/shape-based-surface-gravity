#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit the manuscript against the archived runs.

    cd data && python ../code/audit_manuscript2.py --tex ../manuscript
    ... --values-only     pass 1 only
    ... --quiet           failures and near-misses only

PASS 1 recomputes, from the archived run records, the quantities the paper
states, and compares the two. The expected values written here are what the
manuscript prints; the computed values come from the JSON files. A FAIL means
the manuscript and the records disagree, and one of them is wrong.

PASS 2 works the other way round. It reads the LaTeX sources and, for each
value drawn from the records, reports any number in the text lying close to it
without matching it. That is the signature of a figure written from memory
rather than looked up, which is the error this project met most often.
Near-misses are reported, not failed: most are innocent.

This version matches the manuscript as submitted, in which the scaling
analysis uses the eleven planes below the grain-density ceiling, the stability
ratio is decomposed as 32.1 = 11.78 x 7.52 / 2.77, and every Arrokoth result
except the resolution series excludes the buried facets.

Exit status is 0 if pass 1 is clean.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

XCF_KM = 0.000358        # Itokawa centre of figure, from the shape model
BLOCK_BOUND = 0.1935     # km, plane bound from the block density
GRAIN_BOUND = 0.2069     # km, plane bound from the grain density


def load(name, dirs=(".", "data", "../data")):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
    return None


def qem(records):
    return [r for r in records
            if "meshcache" in r.get("command", "")
            and "--decimator cluster" not in r["command"]]


def run(runs, frag, x=None, mass=None, facets=None, model="head",
        fracsplit=False, scale=False, xtol=1e-4):
    for r in runs:
        if frag not in r["shape_model"]:
            continue
        if ("--scale" in r["command"]) != scale:
            continue
        if ("--fracsplit" in r["command"]) != fracsplit:
            continue
        if mass is not None and abs(r["M_total_kg"] - mass) > 1e-3 * mass:
            continue
        if facets is not None and r["facets_used"] != facets:
            continue
        m = r["models"].get(model)
        if not m or m.get("status") != "ok":
            continue
        if x is not None and abs(m["x_lo_km"] - x) > xtol:
            continue
        return m, r
    return None, None


def sweep(runs):
    out = {}
    for r in runs:
        if "Itokawa" not in r["shape_model"] or r["facets_used"] != 49152:
            continue
        if abs(r["M_total_kg"] - 3.58e10) > 1e7 or "--scale" in r["command"]:
            continue
        m = r["models"].get("head")
        if not m or m["status"] != "ok":
            continue
        mn, dv = m["minimum"], m["derived"]
        out.setdefault(round(m["x_lo_km"], 4), dict(
            f=m["volume_fraction_pct"], rho=mn["rho_R"], rest=mn["rho_rest"],
            drho=mn["rho_R"] - mn["rho_rest"], dM=dv["excess_mass_kg"],
            com=dv["com_offset_m"], V_R=m["V_R_km3"],
            xR=m["region_centroid_x_km"],
            impr=mn["improvement_over_uniform_pct"]))
    return [out[k] for k in sorted(out)], sorted(out)


# The paper computes its sweep statistics from the values as printed in
# Table 3, not from the full-precision records, so that a reader recomputing
# them from the table gets the paper's figures exactly. DIGITS gives the
# precision each column is printed to; the helpers below round to it first.
DIGITS = {"rho": 0, "rest": 1, "drho": 1, "dM": 3, "com": 2, "impr": 3}


def _col(rows, key):
    d = DIGITS.get(key)
    v = np.array([r[key] for r in rows], float)
    if key == "dM":
        v = v / 1e9
    return np.round(v, d) if d is not None else v


def cv(rows, key):
    v = _col(rows, key)
    return 100 * v.std(ddof=1) / abs(v.mean())


def half(rows, key):
    v = _col(rows, key)
    return 100 * (v.max() - v.min()) / 2 / abs(v.mean())


def ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    return b[1], np.sqrt(res @ res / (len(x) - 2) /
                         ((x - x.mean()) ** 2).sum())


class Audit:
    def __init__(self, quiet=False):
        self.n = self.bad = 0
        self.refs = []
        self.quiet = quiet

    def head(self, title):
        if not self.quiet:
            print(f"\n{title}\n{'-' * len(title)}")

    def tol(self, paper):
        # half of the last printed digit, with one exception: ratios of two
        # coefficients of variation inherit the rounding of both columns, so
        # they are checked to the last digit rather than to half of it
        t = repr(float(paper))
        if "e" in t or "E" in t:
            return abs(paper) * 5e-7
        dec = len(t.split(".")[1].rstrip("0")) if "." in t else 0
        return 0.5 * 10 ** (-dec)

    def check(self, label, got, paper, note=""):
        self.n += 1
        slack = 3.0 if "factor" in label or "ratio" in label else 1.0
        ok = got is not None and abs(float(got) - float(paper)) <= \
            slack * self.tol(paper)
        if not ok:
            self.bad += 1
        if got is not None:
            self.refs.append((label, float(paper)))
        if not ok or not self.quiet:
            g = "absent" if got is None else f"{float(got):.6g}"
            print(f"  {'ok  ' if ok else 'FAIL'} {label:<48} "
                  f"paper {float(paper):<11.6g} runs {g}"
                  + (f"   {note}" if note else ""))
        return ok


def pass1(a, rec, syn, el, p4):
    runs = qem(rec)
    print(f"paperV_final.json: {len(rec)} records, {len(runs)} reported")

    a.head("Section 3.1  Itokawa at the detected neck")
    m, r = run(runs, "Itokawa", x=0.1608, facets=49152, mass=3.58e10)
    if m:
        mn, dv = m["minimum"], m["derived"]
        a.check("scan lower limit", r["scan_lo"], 300)
        a.check("scan upper limit", r["scan_hi"], 4000)
        a.check("scan step", r["scan_step"], 25)
        a.check("head density", mn["rho_R"], 2800)
        a.check("background density", mn["rho_rest"], 1858)
        a.check("contrast", mn["contrast"], 1.507)
        a.check("improvement %", mn["improvement_over_uniform_pct"], 16.889)
        a.check("uniform dispersion", r["uniform_reference_dispersion"],
                0.086632)
        a.check("dispersion at the minimum", mn["dispersion"], 0.072000)
        a.check("residual against uniform %",
                100 * mn["dispersion"] / r["uniform_reference_dispersion"],
                83.1)
        a.check("excess mass /1e9 kg", dv["excess_mass_kg"] / 1e9, 2.7525)
        a.check("excess mass % of total", dv["excess_mass_pct_of_total"], 7.7)
        a.check("COM offset m", dv["com_offset_m"], 16.86)
        a.check("region fraction %", m["volume_fraction_pct"], 16.43)
        a.check("region centroid km", m["region_centroid_x_km"], 0.2197)
        a.check("curvature interval", dv["sigma_rho_at_0p60pct"], 129)
        f = m["volume_fraction_pct"] / 100
        a.check("interval propagated to rho_rest",
                f / (1 - f) * round(dv["sigma_rho_at_0p60pct"]), 25.4,
                "the paper propagates the rounded interval, 129")
        a.check("Euler characteristic",
                m.get("manifold", {}).get("euler_characteristic"), 2)
    else:
        print("  FAIL reference run not found")
        a.n += 1
        a.bad += 1

    m, _ = run(runs, "Itokawa", x=0.1608, facets=49152, mass=3.58e10,
               model="neck", xtol=0.06)
    if m:
        a.check("neck model: density", m["minimum"]["rho_R"], 2525)
        a.check("neck model: contrast", m["minimum"]["contrast"], 1.300)
        a.check("neck model: interval",
                m["derived"]["sigma_rho_at_0p60pct"], 259)

    a.head("Section 3.2  resolution stability")
    seen = {}
    for r in runs:
        if "Itokawa" not in r["shape_model"] or "--xsplit" in r["command"]:
            continue
        if "--scale" in r["command"] or abs(r["M_total_kg"] - 3.58e10) > 1e7:
            continue
        h = r["models"].get("head")
        if h and h["status"] == "ok":
            seen[r["facets_used"]] = (h["x_lo_km"], h["minimum"]["rho_R"])
    for fac, wx, wr in ((20000, 0.1702, 2900), (30000, 0.1608, 2800),
                        (40000, 0.1608, 2800), (49152, 0.1608, 2800)):
        g = seen.get(fac)
        a.check(f"{fac} facets: plane km", g[0] if g else None, wx)
        a.check(f"{fac} facets: density", g[1] if g else None, wr)

    rows, planes = sweep(runs)
    at17 = next((rr["rho"] for rr, x in zip(rows, planes)
                 if abs(x - 0.1700) < 1e-4), None)
    a.check("fine mesh at 0.1700 equals the coarse-mesh value", at17, 2900)

    a.head("Section 3.3  the sweep and its three subsets")
    a.check("distinct planes", len(rows), 13)
    subsets = (("thirteen", rows, 4.142, 20.61, 0.251, 2.95, 49.85,
                -1.041, 0.020, -0.177),
               ("eleven", [x for x, p in zip(rows, planes)
                           if p <= GRAIN_BOUND], 1.907, 8.60, 0.268, 3.16,
                23.79, -1.141, 0.018, -0.244),
               ("nine", [x for x, p in zip(rows, planes)
                         if p <= BLOCK_BOUND], 1.677, 6.89, 0.252, 3.00,
                20.13, -1.159, 0.025, -0.262))
    for lab, rr, span, c1, c2, c3, c4, s_, se_, g_ in subsets:
        V = np.array([q["V_R"] for q in rr])
        Vf = np.array([round(q["f"], 2) for q in rr])
        a.check(f"{lab}: V_R span",
                (Vf.max() / Vf.min()) if lab == "eleven"
                else (V.max() / V.min()), span,
                "eleven: from the tabulated fractions")
        a.check(f"{lab}: CV rho_head %", cv(rr, "rho"), c1)
        a.check(f"{lab}: CV rho_rest %", cv(rr, "rest"), c2)
        a.check(f"{lab}: CV excess mass %", cv(rr, "dM"), c3)
        a.check(f"{lab}: CV density difference %", cv(rr, "drho"), c4)
        s, se = ols(np.log(V), np.log(np.abs([q["drho"] for q in rr])))
        g, _ = ols(np.log(V), np.log([q["xR"] - XCF_KM for q in rr]))
        a.check(f"{lab}: scaling exponent", s, s_)
        a.check(f"{lab}: its standard error", se, se_)
        a.check(f"{lab}: geometric slope", g, g_)

    ele = subsets[1][1]
    c = {k: cv(ele, k) for k in ("rho", "rest", "dM", "drho")}
    a.check("eleven: stability ratio", c["rho"] / c["rest"], 32.1)
    a.check("eleven: identity factor", c["dM"] / c["rest"], 11.78,
            "sensitive to the rounding of Delta M in the third decimal")
    a.check("eleven: subtraction factor", c["drho"] / c["rho"], 2.77)
    a.check("eleven: compensation factor", c["drho"] / c["dM"], 7.52)
    a.check("eleven: half-range rho_head %", half(ele, "rho"), 13.20)
    a.check("eleven: half-range rho_rest %", half(ele, "rest"), 0.434)
    a.check("eleven: half-range excess mass %", half(ele, "dM"), 5.12)
    a.check("eleven: half-range COM %", half(ele, "com"), 12.82)
    a.check("eleven: maximum improvement %",
            max(q["impr"] for q in ele), 22.4)
    a.check("thirteen: half-range rho_head %", half(rows, "rho"), 38.26)
    a.check("thirteen: lowest improvement %",
            min(q["impr"] for q in rows), 12.856)
    a.check("thirteen: highest improvement %",
            max(q["impr"] for q in rows), 26.435)
    s11, se11 = ols(np.log([q["V_R"] for q in ele]),
                    np.log(np.abs([q["drho"] for q in ele])))
    a.check("dipole conservation would require", -1 - (-0.244), -0.756)
    a.check("standard errors from that prediction",
            abs(round(s11, 3) + 0.756) / round(se11, 3), 21,
            "from the printed exponent and its printed error")

    a.head("Section 3.7  sensitivity to the assumed mass")
    ms = {}
    for r in runs:
        if "Itokawa" not in r["shape_model"] or "--xsplit" not in r["command"]:
            continue
        if "--scale" in r["command"]:
            continue
        m = r["models"].get("head")
        if m and m["status"] == "ok" and abs(m["x_lo_km"] - 0.1608) < 1e-3:
            ms.setdefault(round(r["bulk_density_kg_m3"]), []).append(
                m["minimum"])
    a.check("distinct assumed densities", len(ms), 5)
    for wb, wr, wc in ((1006, 1325, 1.404), (1510, 2075, 1.484),
                       (2013, 2800, 1.507), (3019, 4275, 1.542),
                       (4026, 5775, 1.569)):
        cand = next((v for k, v in ms.items() if abs(k - wb) <= 2), [])
        best = min(cand, key=lambda z: abs(z["contrast"] - wc)) if cand \
            else None
        a.check(f"bulk {wb}: head density", best["rho_R"] if best else None,
                wr)
        a.check(f"bulk {wb}: contrast", best["contrast"] if best else None,
                wc)
    if len(ms) == 5:
        cs = [min(v, key=lambda z: z["contrast"])["contrast"]
              for v in ms.values()]
        a.check("change over a factor of four %",
                100 * (max(cs) - min(cs)) / min(cs), 11.7)

    a.head("Section 4  the weak-signal cases")
    for lab, frag, x, want in (
            ("Eros at Himeros", "Eros", 4.4698,
             dict(C=1.018, I=0.418, sig=44, f=34.6, com=56.4,
                  disp=0.033992, uni=0.034134)),
            ("Eros misplaced", "Eros", -10.7619,
             dict(C=0.876, I=5.526, sig=15, com=-214.8, f=88.44)),
            ("Bennu", "Bennu", 0.0,
             dict(C=0.999, I=0.000, sig=42, f=50.2, com=-0.07)),
            ("67P", "cg_mspcd", 0.9517,
             dict(C=1.073, I=0.574, f=26.6, com=29.1, sig=28))):
        m, r = run(runs, frag, x=x, xtol=1e-3)
        if not m:
            print(f"  FAIL {lab}: run not found")
            a.n += 1
            a.bad += 1
            continue
        mn, dv = m["minimum"], m["derived"]
        table = dict(C=mn["contrast"], I=mn["improvement_over_uniform_pct"],
                     sig=dv["sigma_rho_at_0p60pct"],
                     f=m["volume_fraction_pct"], com=dv["com_offset_m"],
                     disp=mn["dispersion"],
                     uni=r["uniform_reference_dispersion"])
        for k, v in want.items():
            a.check(f"{lab}: {k}", table[k], v)
        if lab == "Eros misplaced":
            f = m["volume_fraction_pct"] / 100
            a.check("Eros misplaced: displacement statistic",
                    abs(r["bulk_density_kg_m3"] - mn["rho_rest"]) /
                    (f / (1 - f) * round(dv["sigma_rho_at_0p60pct"])), 2.866,
                    "computed from the rounded interval, 15")
            a.check("Eros misplaced: amplification factor",
                    mn["rho_rest"] /
                    (r["bulk_density_kg_m3"] - mn["rho_rest"]), -9.11)
            a.check("Eros misplaced: excess mass % of total",
                    dv["excess_mass_pct_of_total"], -12.33)

    a.head("Section 4.7  Arrokoth")
    clean = load("arrokoth_clean.json")
    if clean:
        a.check("buried facets", clean["buried_facets"], 455)
        a.check("buried area %", clean["buried_area_pct"], 0.943)
    scan = load("arrokoth_scan_clean.json")
    if scan:
        for bulk, wc, wi in (("235", 1.0659, 1.224), ("250", 1.0000, 0.000),
                             ("400", 0.9292, 1.272), ("500", 0.9159, 2.378)):
            cur = scan["curves"].get(bulk)
            a.check(f"Arrokoth at {bulk}: contrast",
                    cur["contrast"] if cur else None, wc)
            a.check(f"Arrokoth at {bulk}: improvement %",
                    cur["improvement_pct"] if cur else None, wi)
    res = {}
    for r in runs:
        if "arrokoth" not in r["shape_model"]:
            continue
        if abs(r["M_total_kg"] - 1.6495e15) > 1e12 or \
                "--fracsplit" in r["command"]:
            continue
        m = r["models"].get("head")
        if m and m["status"] == "ok" and abs(m["x_lo_km"] + 4.7863) < 1e-3:
            res[r["facets_used"]] = (m["minimum"]["rho_R"],
                                     m["minimum"]["rho_rest"],
                                     m["volume_fraction_pct"])
    for fac, wf in ((15000, 66.330), (25000, 66.328), (40960, 66.326)):
        g = res.get(fac)
        a.check(f"Arrokoth {fac} facets: density", g[0] if g else None, 395,
                "this series keeps the buried facets")
        a.check(f"Arrokoth {fac} facets: volume fraction %",
                g[2] if g else None, wf)
    if 40960 in res:
        a.check("Arrokoth resolution series: complement", res[40960][1], 409.8)

    if syn:
        a.head("Section 5  the synthetic family")
        want = {"sphere": (1.24, None, 3.10), "sym_waist": (3.68, None, 5.61),
                "sym_neck": (4.35, None, 6.22), "sym_deep": (4.47, None, 5.60),
                "asym_waist": (16.67, -2.020, 9.28),
                "asym_neck": (17.63, -2.366, 9.37),
                "asym_deep": (19.50, -2.069, 8.81),
                "asym_extreme": (43.71, -1.217, 12.78)}
        for b in syn:
            nm = b.get("name")
            if nm not in want:
                continue
            wi, ws, wr = want[nm]
            a.check(f"{nm}: improvement %", b["improvement_max_pct"], wi)
            a.check(f"{nm}: rho half-range %", b["rho_half_range_pct"], wr)
            if ws is None:
                continue
            pl = b["planes"]
            V = np.array([p["volume_fraction_pct"] for p in pl]) * \
                b["volume_km3"] / 100
            d = np.array([p["rho_R"] - p["rho_rest"] for p in pl])
            s, _ = ols(np.log(V), np.log(np.abs(d)))
            a.check(f"{nm}: exponent", s, ws)
        pts = [(b["improvement_max_pct"], b["scaling_slope"]) for b in syn
               if b.get("scaling_slope") is not None]
        if len(pts) == 4:
            x, y = np.array(pts).T
            a.check("Pearson correlation", np.corrcoef(x, y)[0, 1], 0.95)

    if el:
        a.head("Section 8  the analytic ellipsoid")
        for t, ws, wd in ((2000, 11.62, 0.196), (4000, 4.74, 0.117),
                          (8000, 2.30, 0.049), (16000, 0.93, 0.028),
                          (32000, 0.33, 0.017)):
            s_ = np.array([abs(x["slope_error_pct"]) for x in el
                           if x["target"] == t])
            d_ = np.array([abs(x["dispersion_error_pct"]) for x in el
                           if x["target"] == t])
            a.check(f"{t} facets: slope error %", s_.mean(), ws)
            a.check(f"{t} facets: dispersion error %", d_.mean(), wd)
        for t, wr, wse in ((2000, 59, 18), (32000, 19, 4)):
            s_ = np.array([abs(x["slope_error_pct"]) for x in el
                           if x["target"] == t])
            d_ = np.array([abs(x["dispersion_error_pct"]) for x in el
                           if x["target"] == t])
            ratio = s_.mean() / d_.mean()
            a.check(f"{t} facets: ratio", ratio, wr)
            a.check(f"{t} facets: its standard error",
                    ratio * np.hypot(s_.std(ddof=1) / len(s_) ** .5 / s_.mean(),
                                     d_.std(ddof=1) / len(d_) ** .5 / d_.mean()),
                    wse)
        pair = {}
        for x in el:
            pair.setdefault((tuple(x["axes"]), x["period_h"], x["target"]),
                            {})[x["method"]] = x
        both = [p for p in pair.values() if len(p) == 2]
        dd = np.array([abs(p["qem"]["dispersion_error_pct"]) -
                       abs(p["cluster"]["dispersion_error_pct"])
                       for p in both])
        ds = np.array([abs(p["qem"]["slope_error_pct"]) -
                       abs(p["cluster"]["slope_error_pct"]) for p in both])
        a.check("paired difference, dispersion %", dd.mean(), 0.019)
        a.check("its standard error", dd.std(ddof=1) / len(dd) ** .5, 0.012)
        a.check("paired difference, slope %", ds.mean(), 0.12)
        a.check("its standard error", ds.std(ddof=1) / len(ds) ** .5, 0.20)

    if p4:
        a.head("Appendix D  the mesh-resolution envelope")
        for dec, body, we in (("qem", "Itokawa", 1.35), ("qem", "Eros", 6.03),
                              ("cluster", "Itokawa", 0.95),
                              ("cluster", "Eros", 5.40)):
            v = [x["dispersion"] for x in sorted(
                (z for z in p4 if z["body"] == body and
                 z["decimator"] == dec), key=lambda z: z["facets"])]
            a.check(f"{dec}, {body}: envelope %",
                    100 * (max(v) - min(v)) / min(v), we)
        worst = 0.0
        for dec in ("qem", "cluster"):
            for body in ("Eros", "Itokawa", "Gaspra", "Ida"):
                v = [x["dispersion"] for x in sorted(
                    (z for z in p4 if z["body"] == body and
                     z["decimator"] == dec), key=lambda z: z["facets"])]
                worst = max(worst, 100 * abs(v[-1] - v[-2]) / v[-2])
        a.check("widest plateau between the two finest meshes %", worst, 0.098)

    s2 = load("v9_section2_tests.json")
    if s2:
        a.head("Section 2  the control runs")
        a.check("offset test: change in contrast %", s2["X"], 0.0)
        a.check("offset test: change in improvement, points", s2["Y"], 0.294)
        ax = s2.get("axis", {})
        a.check("uniform body: principal axis, arcsec",
                ax.get("uniform_arcsec"), 2767.6)
        a.check("two-region body: principal axis, arcsec",
                ax.get("two_region_arcsec"), 6729.7)
        a.check("COM from the inertia tensor, m", ax.get("com_x_m"), 17.22)
        sl = s2.get("sliver_max", {})
        if sl:
            a.check("sliver test: change in rho_head", sl.get("rho_head"), 0.0)
            a.check("sliver test: change in rho_rest", sl.get("rho_rest"), 0.0)
    ax2 = load("v9_axis_test.json")
    if ax2:
        a.check("spin axis tilted: contrast", ax2["tilted"]["contrast"], 1.560)
        a.check("spin axis tilted: change %", ax2["contrast_change_pct"], 3.5)
    s8 = load("v10_s8_tests.json")
    if s8 and "omega_control" in s8:
        a.head("Section 8  the omega = 0 control")
        off = s8["omega_control"]["without_rotation"]
        on = s8["omega_control"]["with_rotation"]
        a.check("without rotation: head density", off["rho_R"], 2950)
        a.check("without rotation: background density", off["rho_rest"],
                1828.6)
        a.check("without rotation: contrast", off["contrast"], 1.613)
        a.check("without rotation: improvement %", off["improvement"], 17.16)
        a.check("change in contrast %",
                100 * (off["contrast"] / on["contrast"] - 1), 7.1)
        a.check("change in background density",
                on["rho_rest"] - off["rho_rest"], 29.5)
        for b, wr in (("itokawa", 0.0695), ("eros", 0.3076),
                      ("bennu", 0.5857), ("cg", 0.2114),
                      ("arrokoth", 0.2105)):
            g = s8.get("geometry", {}).get(b)
            a.check(f"{b}: centrifugal over gravitational at the equator",
                    g["a_over_g"] if g else None, wr)

    a.head("Section 2  the shape models")
    for lab, frag, wf, wv, wchi in (
            ("Itokawa", "Itokawa", 49152, 0.0177856, 2),
            ("Eros", "Eros", 49152, 2506.39, 2),
            ("Bennu", "Bennu", 50000, 0.061553, 2),
            ("67P", "cg_mspcd", 50000, 18.5912, 2),
            ("Arrokoth", "arrokoth", 40960, 4123.81, 4)):
        m, r = run(runs, frag)
        if not r:
            continue
        a.check(f"{lab}: facets used", r["facets_used"], wf)
        a.check(f"{lab}: volume", r["volume_km3"], wv)
        a.check(f"{lab}: Euler characteristic",
                m.get("manifold", {}).get("euler_characteristic"), wchi)


NUM = re.compile(r"(?<![\w.])(\d{1,7}(?:[.,]\d+)?)(?![\w])")


def pass2(a, texdir):
    files = sorted(glob.glob(os.path.join(texdir, "*.tex")))
    if not files:
        print(f"\nno .tex files in {texdir}; pass 2 skipped")
        return
    print(f"\n{'=' * 78}\nPASS 2: near-misses in {len(files)} LaTeX files\n"
          f"{'=' * 78}")
    found = {}
    for path in files:
        text = re.sub(r"(?<!\\)%.*", "",
                      open(path, encoding="utf-8", errors="replace").read())
        for mt in NUM.finditer(text):
            try:
                v = float(mt.group(1).replace(",", ""))
            except ValueError:
                continue
            found.setdefault(v, []).append(
                (os.path.basename(path), text.count("\n", 0, mt.start()) + 1))

    def rounded_form(v, ref):
        """Is v just ref quoted to fewer digits? Those are not errors."""
        return any(abs(v - round(ref, k)) < 1e-9 for k in range(0, 7))

    def sig(x):
        t = f"{abs(x):.10g}".replace(".", "").lstrip("0")
        return len(t.rstrip("0")) or 1

    hits = 0
    for label, ref in sorted(a.refs, key=lambda t: t[0]):
        # a value with one or two significant digits has neighbours
        # everywhere and flagging them buries the ones that matter
        if ref == 0 or sig(ref) < 3:
            continue
        tol = a.tol(ref)
        # if the paper states the value, in any rounding, its neighbours are
        # other quantities and not mistakes. Only a value the paper never
        # states, sitting beside one it does, is worth a reader's time.
        if any(abs(v - ref) <= tol or rounded_form(v, ref) for v in found):
            continue
        near = [(v, abs(v - ref) / abs(ref), w) for v, w in found.items()
                if abs(v - ref) > tol and abs(v - ref) / abs(ref) < 0.012
                and sig(v) >= 3 and not rounded_form(v, ref)]
        if not near:
            continue
        near.sort(key=lambda t: t[1])
        hits += 1
        print(f"  {label}  = {ref:.6g}")
        for v, rel, where in near[:3]:
            loc = ", ".join(f"{f}:{l}" for f, l in where[:3])
            more = f" and {len(where) - 3} more" if len(where) > 3 else ""
            print(f"      {v:<12.6g} {100 * rel:5.2f}% away   {loc}{more}")
    if not hits:
        print("  none found within six per cent of any reference value.")
    else:
        print(f"\n  {hits} reference values have a close neighbour in the "
              f"text, within 1.2 per cent\n  and quoted to three digits or "
              f"more. Read them: a number that should be one\n  of these and "
              f"is not is the error this pass looks for.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tex", default="../manuscript")
    ap.add_argument("--values-only", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rec = load("paperV_final.json")
    if rec is None:
        sys.exit("cannot find paperV_final.json; run this from data/")
    syn, el, p4 = (load("synth_final.json"), load("ellipsoid_test.json"),
                   load("paper4_recheck.json"))
    for nm, obj in (("synth_final.json", syn), ("ellipsoid_test.json", el),
                    ("paper4_recheck.json", p4)):
        if obj is None:
            print(f"note: {nm} not found; those checks are skipped")

    a = Audit(quiet=args.quiet)
    pass1(a, rec, syn, el, p4)
    print(f"\n{'=' * 78}")
    print(f"PASS 1: {a.n - a.bad} of {a.n} checks passed")
    if a.bad:
        print(f"{a.bad} FAILED — the manuscript and the records disagree")
    else:
        print("every checked value is reproduced by the archived records")
    if not args.values_only:
        pass2(a, args.tex)
    return 1 if a.bad else 0


if __name__ == "__main__":
    sys.exit(main())
