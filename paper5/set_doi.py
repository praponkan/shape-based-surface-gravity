#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insert the reserved DOIs into every file that carries a placeholder.

    python set_doi.py --deposit 10.5281/zenodo.1234567 \
                      --paper   10.5281/zenodo.7654321

Zenodo lets a DOI be reserved before publication, so both identifiers can be
known in advance. Run this from the top of the package, once, after reserving
them and before uploading. It rewrites in place and reports what it changed.

Two DOIs are involved and they are not the same thing:

    --deposit   this data-and-code archive
    --paper     the manuscript, if it is deposited separately

If the manuscript and this archive share one deposit, pass the same value to
both; the script will say so rather than pretending they differ.

    python set_doi.py --check     report which placeholders remain, change
                                  nothing, and exit non-zero if any are left
"""

import argparse
import os
import re
import sys

PLACEHOLDER = "10.5281/zenodo.XXXXXXXX"

# file -> what the DOI in it refers to
TARGETS = {
    ".zenodo.json": "paper",
    "CITATION.cff": "deposit",
    "README.md": "deposit",
    os.path.join("manuscript", "Section9.tex"): "deposit",
}


def bare(doi):
    """Accept a full URL, a doi: prefix, or the bare identifier."""
    d = doi.strip()
    for p in ("https://doi.org/", "http://doi.org/", "doi:", "DOI:"):
        if d.startswith(p):
            d = d[len(p):]
    return d.strip()


def scan(root):
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
        for fn in filenames:
            if fn.endswith((".pdf", ".zip", ".npz", ".png")):
                continue
            if fn == os.path.basename(__file__):
                continue   # this script contains the placeholder by design
            path = os.path.join(dirpath, fn)
            try:
                text = open(path, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            n = text.count("XXXXXXXX")
            if n:
                found[os.path.relpath(path, root)] = n
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deposit", help="DOI of this data-and-code archive")
    ap.add_argument("--paper", help="DOI of the manuscript")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--check", action="store_true",
                    help="report remaining placeholders and exit")
    a = ap.parse_args()

    if a.check or not (a.deposit or a.paper):
        found = scan(a.root)
        if not found:
            print("No placeholders remain.")
            return 0
        print("Placeholders still present:")
        for f, n in sorted(found.items()):
            print(f"  {f}  ({n})")
        if not a.check:
            print("\nRun with --deposit and --paper to fill them in.")
        return 1

    dep = bare(a.deposit) if a.deposit else None
    pap = bare(a.paper) if a.paper else None
    if dep and pap and dep == pap:
        print(f"Both DOIs are {dep}; the manuscript and the archive are one "
              f"deposit.")
    for label, val in (("deposit", dep), ("paper", pap)):
        if val and not re.match(r"^10\.\d{4,9}/\S+$", val):
            sys.exit(f"--{label} does not look like a DOI: {val}")

    changed = 0
    for rel, kind in TARGETS.items():
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            print(f"  skipped, not present: {rel}")
            continue
        val = dep if kind == "deposit" else pap
        if not val:
            print(f"  skipped, no --{kind} given: {rel}")
            continue
        text = open(path, encoding="utf-8").read()
        n = text.count("XXXXXXXX")
        if not n:
            print(f"  no placeholder in: {rel}")
            continue
        text = text.replace(PLACEHOLDER, val)
        text = text.replace("XXXXXXXX", val.split(".")[-1])
        open(path, "w", encoding="utf-8").write(text)
        print(f"  {rel}: {n} replaced with the {kind} DOI, {val}")
        changed += n

    print(f"\n{changed} placeholders filled.")
    left = scan(a.root)
    if left:
        print("Still remaining:")
        for f, n in sorted(left.items()):
            print(f"  {f}  ({n})")
        return 1
    print("None remain. Recompile the manuscript before uploading, since "
          "Section9 changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
