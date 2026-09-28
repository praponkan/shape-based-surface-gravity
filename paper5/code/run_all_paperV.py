#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper V, complete run set: one command, every number in the paper.

    python run_all_paperV.py

Every result quoted in the manuscript is regenerated here with the current
solver, so that the archived record matches the released code exactly. The
numbers are expected to be unchanged; the point is reproducibility, not
rediscovery.

Fifty-three jobs in eight groups:

    A   Itokawa split-plane sweep                     12
    B   Itokawa resolution series                      4
    C   Itokawa mass sensitivity                       5
    D   Itokawa scale test                             1
    E   Eros                                           2
    F   Bennu                                          1
    G   67P                                            9
    H   Arrokoth                                      13
    S   synthetic family (separate script)             8 bodies

Expect six to eight hours. The batch is resumable: rerun the same command after
an interruption and it will skip what is already recorded. A failed job is
reported and does not stop the rest.

Options
-------
    --record FILE    output JSON (default paperV_final.json)
    --script FILE    solver (default density_estimate.py)
    --synth FILE     synthetic driver (default synth_invariance.py)
    --synth-record   synthetic output (default synth_final.json)
    --only PREFIX    run only jobs whose label starts with PREFIX, e.g. G
    --skip-synth     omit the synthetic family
    --list           print the job list and exit
    --dry-run        print the commands without running
    --force          ignore the record and rerun everything
    --no-selftest    skip the geometry self-test
    --compare FILE   after running, compare against a previous record
    --summary        summarise an existing record and exit
"""

import json
import os
import subprocess
import sys
import time

RECORD = "paperV_final.json"
MESHCACHE = "meshcache"
SCRIPT = "density_estimate.py"
SYNTH = "synth_invariance.py"
SYNTH_RECORD = "synth_final.json"

ITOKAWA = "Itokawa Hayabusa 50k poly.obj"
EROS = "Eros Gaskell 50k poly.obj"
BENNU = "Bennu_v20_200k.obj"
CG = "cg_mspcd_shap2_001m_cart.obj"
CG_SPC = "Churyumov-Gerasimenko SPC 2017 - 199k poly.obj"
ARROKOTH = "arrokoth_porter_2024_v01.obj"

M_ITO, P_ITO = "3.58e10", "12.1324"
M_EROS, P_EROS = "6.687e15", "5.27"
M_BENNU, P_BENNU = "7.330e10", "4.296061"
M_CG, P_CG = "9.982e12", "12.4043"
P_ARR = "15.92"
M_ARR = {250: "1.0310e15", 400: "1.6495e15", 500: "2.0619e15"}

JOBS = []


def job(label, args, purpose):
    JOBS.append({"label": label, "args": args, "purpose": purpose})


def base(obj, mtotal, period, target="50000"):
    return [obj, "--mtotal", mtotal, "--period", period, "--target", target,
            "--meshcache", MESHCACHE]


# ---- A. Itokawa split-plane sweep, twelve planes ------------------------
# 0.1608 is the detected neck; the rest are set by hand. Scan widened from
# 0.200 onward because rho_head passes the original 3900 ceiling there.
for i, (x, scan) in enumerate([
        (0.130, "300,4000,25"), (0.140, "300,4000,25"), (0.150, "300,4000,25"),
        (0.160, "300,4000,25"), (0.170, "300,4000,25"), (0.180, "300,4000,25"),
        (0.185, "300,8000,25"), (0.190, "300,8000,25"), (0.195, "300,8000,25"),
        (0.200, "300,8000,25"), (0.220, "300,8000,25"), (0.240, "300,8000,25"),
], start=1):
    job(f"A{i:02d}",
        base(ITOKAWA, M_ITO, P_ITO) + ["--model", "head",
                                       "--xsplit", f"{x:.3f}", "--scan", scan],
        f"split-plane sweep at x = {x:.3f} km")

# the reference run: both models at the detected neck, plane left automatic
job("A13", base(ITOKAWA, M_ITO, P_ITO) + ["--scan", "300,4000,25"],
    "reference run at the detected neck, head and neck models")

# ---- B. Itokawa resolution series ---------------------------------------
for i, t in enumerate(["20000", "30000", "40000", "50000"], start=1):
    job(f"B{i}", base(ITOKAWA, M_ITO, P_ITO, target=t) +
        ["--scan", "300,4000,25"],
        f"resolution series, decimation target {t}")

# ---- C. Itokawa mass sensitivity ----------------------------------------
for i, (m, scan) in enumerate([
        ("1.79e10", "150,2000,25"), ("2.685e10", "200,3000,25"),
        ("3.58e10", "300,4000,25"), ("5.37e10", "450,6000,25"),
        ("7.16e10", "600,8000,25")], start=1):
    job(f"C{i}", base(ITOKAWA, m, P_ITO) +
        ["--model", "head", "--xsplit", "0.1608", "--scan", scan],
        f"mass sensitivity, M = {m} kg")

# ---- D. Itokawa scale test ----------------------------------------------
job("D1", base(ITOKAWA, M_ITO, P_ITO) +
    ["--model", "head", "--xsplit", "0.1627", "--scale", "1.011385",
     "--scan", "300,4000,25"],
    "mesh enlarged to the published volume, plane moved with it")

# ---- E. Eros -------------------------------------------------------------
job("E1", base(EROS, M_EROS, P_EROS) +
    ["--model", "head", "--xsplit", "4.47", "--scan", "500,5000,25"],
    "Eros at the true saddle, Himeros")
job("E2", base(EROS, M_EROS, P_EROS) +
    ["--model", "head", "--xsplit", "-10.7619", "--scan", "500,5000,25"],
    "Eros at the misplaced plane, for comparison")

# ---- F. Bennu ------------------------------------------------------------
job("F1", base(BENNU, M_BENNU, P_BENNU) +
    ["--model", "head", "--xsplit", "0.0", "--scan", "200,3000,10"],
    "Bennu, no saddle exists, plane imposed at x = 0")

# ---- G. 67P --------------------------------------------------------------
job("G1", base(CG, M_CG, P_CG) + ["--scan", "100,4000,5"],
    "67P reference at the detected neck, head and neck models")
for i, x in enumerate([0.70, 0.80, 0.90, 1.00, 1.10, 1.20], start=2):
    job(f"G{i}", base(CG, M_CG, P_CG) +
        ["--model", "head", "--xsplit", f"{x:.2f}", "--scan", "100,4000,5"],
        f"67P split-plane sweep at x = {x:.2f} km")
job("G8", base(CG_SPC, M_CG, P_CG) + ["--scan", "100,4000,5"],
    "67P on the second shape model, cross-version check")
job("G9", base(CG, M_CG, P_CG) +
    ["--model", "head", "--xsplit", "0.947", "--cliptol", "1e-6",
     "--scan", "100,4000,5"],
    "67P with the clipping tolerance raised, sliver control")

# ---- H. Arrokoth ---------------------------------------------------------
for i, rho in enumerate([250, 400, 500], start=1):
    job(f"H{i}", base(ARROKOTH, M_ARR[rho], P_ARR) + ["--scan", "25,3000,5"],
        f"Arrokoth at the detected neck, assumed bulk {rho} kg/m3")
for i, f in enumerate([0.400, 0.445, 0.495, 0.550, 0.612, 0.680, 0.756],
                      start=4):
    job(f"H{i:02d}", base(ARROKOTH, M_ARR[400], P_ARR) +
        ["--model", "head", "--fracsplit", f"{f:.3f}", "--scan", "25,3000,5"],
        f"Arrokoth sweep, region {100*f:.1f}% of the volume")
for i, t in enumerate(["15000", "25000", "50000"], start=11):
    job(f"H{i}", base(ARROKOTH, M_ARR[400], P_ARR, target=t) +
        ["--model", "head", "--xsplit", "-4.7863", "--scan", "25,3000,5"],
        f"Arrokoth resolution series, target {t} facets")


# ---- I. decimator comparison, on the three bodies that are decimated -----
# Same plane, same mass, same target: the only difference is how the mesh was
# reduced. This measures what changing the decimator does to the answer, which
# is the question that bears on Papers III and IV as well as this one.
job("I1", base(ITOKAWA, M_ITO, P_ITO, target="40000") +
    ["--decimator", "cluster", "--model", "head", "--xsplit", "0.1608",
     "--scan", "300,4000,25"],
    "Itokawa at 40k by vertex clustering, for comparison with B3")
job("I2", base(BENNU, M_BENNU, P_BENNU) +
    ["--decimator", "cluster", "--model", "head", "--xsplit", "0.0",
     "--scan", "200,3000,10"],
    "Bennu by vertex clustering, for comparison with F1")
job("I3", base(CG, M_CG, P_CG) +
    ["--decimator", "cluster", "--model", "head", "--xsplit", "0.947",
     "--scan", "100,4000,5"],
    "67P by vertex clustering, for comparison with G1")


# --------------------------------------------------------------------------
def already_done(record, script, args):
    if not os.path.exists(record):
        return False
    try:
        data = json.load(open(record, encoding="utf-8"))
    except Exception:
        return False
    want = " ".join([script] + args + ["--record", record])
    return any(r.get("command") == want for r in data)


def summarise(record):
    if not os.path.exists(record):
        print("no record file to summarise")
        return
    data = json.load(open(record, encoding="utf-8"))
    print(f"\n{'='*100}\n{record}: {len(data)} runs\n{'='*100}")
    hdr = (f"{'body':<24} {'dec':<4} {'model':<5} {'x_lo':>9} {'reg%':>6} "
           f"{'rho_R':>8} {'rho_rest':>9} {'contr':>7} {'impr%':>8} "
           f"{'chi':>4} {'int':>4}")
    print(hdr + "\n" + "-" * len(hdr))
    edge, bad, nonmf = [], [], []
    for r in data:
        b = os.path.basename(r.get("shape_model", "?"))[:24]
        dec = ("clus" if "--decimator cluster" in r.get("command", "")
               else ("qem" if r.get("facets_native") != r.get("facets_used")
                     else "full"))
        for nm, m in r.get("models", {}).items():
            if m.get("status") != "ok" or "minimum" not in m:
                bad.append((b, nm, m.get("status")))
                print(f"{b:<24} {dec:<4} {nm:<5} {m.get('status','?')}")
                continue
            mn = m["minimum"]
            chi = m.get("manifold", {}).get("euler_characteristic", "-")
            if chi != 2:
                nonmf.append((b, nm, chi))
            if not mn["interior_minimum"]:
                edge.append((b, nm))
            print(f"{b:<24} {dec:<4} {nm:<5} {m['x_lo_km']:>9.4f} "
                  f"{m['volume_fraction_pct']:>6.1f} {mn['rho_R']:>8.1f} "
                  f"{mn['rho_rest']:>9.1f} {mn['contrast']:>7.4f} "
                  f"{mn['improvement_over_uniform_pct']:>8.3f} {str(chi):>4} "
                  f"{'yes' if mn['interior_minimum'] else 'EDGE':>4}")
    if nonmf:
        print("\nNON-MANIFOLD REGIONS (chi should be 2):")
        for b, nm, chi in nonmf:
            print(f"  {b}  {nm}: chi = {chi}")
    if edge:
        print("\nEdge minima, to be discarded or rerun with a wider scan:")
        for b, nm in edge:
            print(f"  {b}  {nm}")
    if bad:
        print("\nAborted models:")
        for b, nm, st in bad:
            print(f"  {b}  {nm}: {st}")

    # side-by-side where both decimators were run on the same configuration
    pairs = {}
    for r in data:
        clus = "--decimator cluster" in r.get("command", "")
        for nm, m in r.get("models", {}).items():
            if m.get("status") != "ok" or nm != "head":
                continue
            k = (os.path.basename(r["shape_model"]), round(m["x_lo_km"], 3),
                 r["facets_used"] if not clus else None)
            kk = (k[0], k[1])
            pairs.setdefault(kk, {})["cluster" if clus else "qem"] = (m, r)
    both = {k: v for k, v in pairs.items() if len(v) == 2}
    if both:
        print(f"\n{'='*100}\nDECIMATOR COMPARISON: quadric collapse against "
              f"vertex clustering\n{'='*100}")
        print(f"{'body':<24} {'x':>8} | {'faces':>7} {'chi':>4} {'vol':>10} "
              f"{'rho_R':>7} {'contr':>7} {'impr%':>7}")
        for k, v in sorted(both.items()):
            for lab in ("qem", "cluster"):
                m, r = v[lab]
                chi = m.get("manifold", {}).get("euler_characteristic", "-")
                print(f"{(k[0][:22] + ' ' + lab):<24} {k[1]:>8.3f} | "
                      f"{r['facets_used']:>7} {str(chi):>4} "
                      f"{r['volume_km3']:>10.5f} "
                      f"{m['minimum']['rho_R']:>7.1f} "
                      f"{m['minimum']['contrast']:>7.4f} "
                      f"{m['minimum']['improvement_over_uniform_pct']:>7.3f}")
            mq, mc = v["qem"][0], v["cluster"][0]
            dc = 100 * (mq["minimum"]["contrast"] /
                        mc["minimum"]["contrast"] - 1)
            print(f"{'  -> contrast differs by':<24} {dc:>8.3f}%")


def compare(new, old):
    """Compare two record files run by run, on the quantities the paper uses."""
    if not (os.path.exists(new) and os.path.exists(old)):
        print(f"cannot compare: need both {new} and {old}")
        return
    A = json.load(open(old, encoding="utf-8"))
    B = json.load(open(new, encoding="utf-8"))

    def key(r, nm, m):
        return (os.path.basename(r["shape_model"]), nm,
                round(m["x_lo_km"], 4), round(r["M_total_kg"], 6),
                r["facets_used"])

    def index(data):
        out = {}
        for r in data:
            for nm, m in r.get("models", {}).items():
                if m.get("status") == "ok" and "minimum" in m:
                    out[key(r, nm, m)] = (m, r)
        return out

    ia, ib = index(A), index(B)
    shared = sorted(set(ia) & set(ib))
    print(f"\n{'='*96}\nCOMPARISON: {new} against {old}\n{'='*96}")
    print(f"  {len(ia)} models in the old record, {len(ib)} in the new, "
          f"{len(shared)} in common")
    fields = [("rho_R", lambda m: m["minimum"]["rho_R"]),
              ("rho_rest", lambda m: m["minimum"]["rho_rest"]),
              ("contrast", lambda m: m["minimum"]["contrast"]),
              ("improvement", lambda m: m["minimum"]
               ["improvement_over_uniform_pct"]),
              ("excess_mass", lambda m: m["derived"]["excess_mass_kg"]),
              ("com_offset", lambda m: m["derived"]["com_offset_m"])]
    worst = {f: (0.0, None) for f, _ in fields}
    for k in shared:
        ma, mb = ia[k][0], ib[k][0]
        for f, get in fields:
            try:
                va, vb = get(ma), get(mb)
            except Exception:
                continue
            if va in (0, None) or vb is None:
                continue
            d = abs(100 * (vb / va - 1))
            if d > worst[f][0]:
                worst[f] = (d, k)
    print(f"\n  largest relative change per quantity:")
    for f, _ in fields:
        d, k = worst[f]
        where = f"{k[0]} {k[1]} x={k[2]}" if k else "-"
        print(f"    {f:<12} {d:9.6f}%   {where}")
    big = [f for f, _ in fields if worst[f][0] > 0.01]
    print()
    if big:
        print(f"  CHANGED beyond 0.01%: {', '.join(big)}. Inspect before "
              f"updating the manuscript.")
    else:
        print("  Every quantity agrees to better than 0.01%. The recorded "
              "values stand.")
    missing = sorted(set(ia) - set(ib))
    if missing:
        print(f"\n  {len(missing)} models present in the old record but not "
              f"the new:")
        for k in missing[:10]:
            print(f"    {k[0]} {k[1]} x={k[2]}")


def main():
    argv = sys.argv[1:]

    def opt(n, d=None):
        return argv[argv.index(n) + 1] if n in argv else d

    record = opt("--record", RECORD)
    script = opt("--script", SCRIPT)
    synth = opt("--synth", SYNTH)
    synth_rec = opt("--synth-record", SYNTH_RECORD)
    only = opt("--only")
    jobs = [j for j in JOBS if not only or j["label"].startswith(only)]

    if "--list" in argv:
        for j in jobs:
            print(f"  {j['label']:<5} {' '.join(j['args'])}")
        print(f"\n  {len(jobs)} solver jobs" +
              ("" if "--skip-synth" in argv else ", plus the synthetic family"))
        return
    if "--summary" in argv:
        summarise(record)
        return
    if "--compare" in argv and "--dry-run" in argv:
        compare(record, opt("--compare"))
        return
    if not os.path.exists(script):
        sys.exit(f"solver not found: {script}")

    if "--no-selftest" not in argv and "--dry-run" not in argv:
        print("Geometry self-test ...")
        if subprocess.run([sys.executable, script, "--selftest"]).returncode:
            sys.exit("SELF-TEST FAILED. Fix this before running the batch.")
        print()

    force, dry = "--force" in argv, "--dry-run" in argv
    t0all = time.time()
    done = skipped = failed = 0
    failures = []

    for i, j in enumerate(jobs, start=1):
        cmd = [sys.executable, script] + j["args"] + ["--record", record]
        head = f"[{i}/{len(jobs)}] {j['label']}"
        if dry:
            print(f"{head}  {' '.join(cmd[1:])}")
            continue
        if not force and already_done(record, script, j["args"]):
            print(f"{head}  already recorded, skipping")
            skipped += 1
            continue
        print(f"\n{'='*78}\n{head}  {j['purpose']}\n"
              f"  {' '.join(j['args'])}\n{'='*78}")
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        dt = (time.time() - t0) / 60
        if rc == 0:
            done += 1
            print(f"  [{j['label']} finished in {dt:.1f} min]")
        else:
            failed += 1
            failures.append((j["label"], rc))
            print(f"  [{j['label']} FAILED with code {rc}; continuing]")
        el = (time.time() - t0all) / 60
        left = len(jobs) - i
        if done and left:
            print(f"  elapsed {el:.0f} min, about {el/done*left:.0f} min left "
                  f"for {left} more")

    if dry:
        return

    if "--skip-synth" not in argv and not only:
        if os.path.exists(synth):
            print(f"\n{'='*78}\nSynthetic family\n{'='*78}")
            t0 = time.time()
            rc = subprocess.run([sys.executable, synth,
                                 "--record", synth_rec]).returncode
            print(f"  [synthetic family {'finished' if rc == 0 else 'FAILED'} "
                  f"in {(time.time()-t0)/60:.1f} min]")
            if rc:
                failures.append(("synthetic", rc))
        else:
            print(f"\n  {synth} not found; synthetic family skipped")

    print(f"\n{'='*78}\nBATCH DONE in {(time.time()-t0all)/60:.0f} min: "
          f"{done} run, {skipped} skipped, {failed} failed")
    for lab, rc in failures:
        print(f"  failed: {lab} (exit {rc})")
    summarise(record)
    if "--compare" in argv:
        compare(record, opt("--compare"))


if __name__ == "__main__":
    main()
