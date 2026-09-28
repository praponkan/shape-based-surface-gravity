# What potential-dispersion minimization constrains in small-body interiors

Paper V of the shape-based surface gravity series. This archive holds the
manuscript sources, the analysis code, the archived run records and the
figures, and it is the release that produced every number in the paper.

    doi:10.5281/zenodo.23003040          CC BY 4.0

Shape models are **not** redistributed; `data/SHAPE_MODELS.md` gives their
sources.

## Layout

```
manuscript/  main.tex, eleven sections, four appendices
code/        23 scripts: the solver, the decimator, the generators,
             the tests, the runners, the figure scripts and the audits
data/        18 files: the archived runs, the master value file and three notes
figures/     the eight figures as they appear in the paper
set_doi.py   fills DOI placeholders (already filled in this release)
DEPOSIT_CHECKLIST.md
```

## Checking the paper against the records

```
cd data && python ../code/audit_manuscript.py --tex ../manuscript
```

Pass 1 recomputes 202 quantities from the archived records and compares them
with what the manuscript prints, to half the last printed digit. Where the
paper computes a statistic from its own tables rather than from full precision
— the sweep statistics of Table 3 are the case — the script follows that
convention, so a reader recomputing from the printed table gets the paper's
figures exactly. Pass 2 reads the LaTeX and reports any value the paper never
states that has a close neighbour it does state, which is the signature of a
figure written from memory rather than looked up. Neither pass needs a shape
model or a gravity library, and both run in a second.

All 202 checks pass on the manuscript as deposited. Pass 2 reports two
near-misses, both page numbers in the bibliography.

`code/check_key_results.py` is the older and narrower version of the same idea
and is kept because the paper cites it.

## Reproducing the figures

```
cd data
python ../code/make_figs.py          # figures 1, 6, 7
python ../code/make_figs2.py         # figure 4
python ../code/make_fig03.py         # figure 3
python ../code/make_fig05.py         # figure 5
python ../code/make_fig08.py         # figure 8
```

Needs only `numpy` and `matplotlib`. Figure 2 comes from `run_profiles.py`,
which needs the shape models.

## Reproducing the results

The inversions need the shape models and a polyhedron gravity binding
(`polyhedral-gravity` 3.3.1 was used). A full rerun takes about fourteen
hours:

```
python code/run_all_paperV.py --meshcache meshcache
```

Two tests need no shape model at all:

```
python code/test_ellipsoid.py            # the analytic ellipsoid
python code/decimate_qem.py --selftest   # the decimator
```

`test_ellipsoid.py` takes `--subdiv` and `--targets`. The 49,152-facet row
discussed in Section 8 needs `--subdiv 7`, which is about 256 times the work of
the default and was abandoned for that reason; one configuration completed and
is reported in the text.

## What each script does

| Script | Purpose |
|---|---|
| `density_estimate.py` | the region solver: plane clipping, the geometric checks, the density scan |
| `decimate_qem.py` | quadric edge collapse with the link condition; preserves the manifold property |
| `surface_slope_lib.py` | Paper IV's field routine with a decimator dispatcher added |
| `synth_bodies.py`, `synth_invariance.py` | the eight synthetic contact binaries and their sweeps |
| `test_ellipsoid.py` | the analytic ellipsoid: settles the Paper IV comparison without assuming any mesh is the truth |
| `check_arrokoth.py` | intersection volume and buried facets in a multi-component model |
| `rerun_arrokoth_clean.py` | Arrokoth with the buried facets excluded |
| `arrokoth_scan_clean.py` | the same run with the whole dispersion curve archived, for figure 3 |
| `planted_anomaly.py`, `diagnose_planted.py` | the relaxation experiment of Section 8, which did not converge |
| `v9_section2_tests.py` | field-point offset sensitivity, principal-axis rotation, sliver test |
| `v9_axis_test.py` | the inversion with the spin axis taken as the recovered principal axis |
| `v10_s8_tests.py` | the omega = 0 control and the equatorial geometry of all five bodies |
| `run_all_paperV.py`, `run_profiles.py` | batch runners |
| `make_fig*.py`, `make_figs*.py` | figures |
| `audit_manuscript.py`, `check_key_results.py` | verification against the archived records |

## Data files

| File | Contents |
|---|---|
| `all_results.json` | the consolidated record, including superseded values and withdrawn hypotheses |
| `PaperV_final_numbers.md` | the master value file. **Read section 0 first**: earlier sections predate the final reruns and section 12 is authoritative where they disagree |
| `paperV_final.json` | the run records; those carrying `--meshcache` are the ones the paper reports |
| `synth_final.json` | the eight synthetic bodies, with per-plane results |
| `ellipsoid_test.json` | ninety analytic-ellipsoid configurations |
| `paper4_recheck.json` | the Paper IV resolution ladder recomputed with both decimators; Appendix D is built from it |
| `arrokoth_clean.json`, `arro_235/250/500.json` | Arrokoth with and without the buried facets, at four assumed masses |
| `arrokoth_scan_clean.json` | the full dispersion curves behind figure 3 |
| `v9_section2_tests.json`, `v9_axis_test.json`, `v10_s8_tests.json` | the control runs reported in Sections 2 and 8 |
| `planted_anomaly.json`, `diagnose_planted.json` | the relaxation attempt |
| `SHAPE_MODELS.md` | sources, volumes, and three properties worth knowing before reuse |
| `DECIMATION_NOTES.txt` | the limitations of vertex clustering, written as a supplement to Paper IV |

## Environment

Results were computed under Windows 10 (build 19045) with Python 3.13 and
`polyhedral-gravity` 3.3.1. The geometry self-test reproduces its printed
values digit for digit under Ubuntu 24.04 with Python 3.12.3. See
`requirements.txt`.
