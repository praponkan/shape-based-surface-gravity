# SDGA Paper IV — Code and Data Package

Supporting material for:

**Spatial surface gravity of small bodies from polyhedral shape models:
slope, geopotential, and the resolution robustness of geopotential dispersion**
Prapon Kanjanatarayont (Independent Researcher, Thailand)
ORCID 0009-0009-4081-7422 · prapon.kanjana@gmail.com

Archived release: doi:10.5281/zenodo.21793575

Companion papers in the SDGA series:
- Paper I   — doi:10.5281/zenodo.21327611
- Paper II  — doi:10.5281/zenodo.21328178
- Paper III — doi:10.5281/zenodo.21622962

Repository: https://github.com/praponkan/shape-based-surface-gravity

---

## 1. What this package contains

Everything needed to reproduce the numerical results of Paper IV: the analysis
code, two independent verification scripts against exact analytic solutions,
mesh diagnostics, the five figures, the complete numerical results in
machine-readable form, and the full manuscript source.

```
code/           analysis programs used to produce the paper's numbers
verification/   checks against exact analytic solutions
diagnostics/    mesh and axis checks run before any analysis
figures/        make_paper4_figures.py and the five figure PDFs
results/        all_results.json  — every number in the paper
manuscript/     complete LaTeX source: main.tex, Sections 1-8,
                Appendix A, figures.tex, data_availability.tex
```

## 2. Requirements

- Python 3.10+
- numpy, scipy
- `polyhedral-gravity` v3.3.1 (analytic polyhedral solver of Werner & Scheeres 1996):
  `pip install polyhedral-gravity`

Shape models are NOT redistributed here. Sources are listed in Section 5.

## 3. Reproducing the results

Verification first (both must pass before trusting any output):

```
python verification/rotsphere_check.py      # slope field vs analytic rotating sphere
python verification/potential_check.py      # potential vs analytic ellipsoid
python code/surface_slope_lib.py selftest   # non-rotating sphere control
```

Validation:

```
# Ryugu, per-facet against the Hayabusa2 archive
python code/ryugu_facet_validation.py "Ryugu_SHAPE_SFM_49k_v20180804.obj" \
       "Gravity_SHAPE_SFM_49k_v20180804_7.63_1200a.txt"

# Bennu, slope vs assumed bulk density
python code/density_sweep.py "Bennu_v20_200k.obj" 4.296061 --target 30000
```

Surface fields for a single body:

```
python code/surface_slope_lib.py <obj> <density_g_cm3> <period_h> [--target N] [--both|--nospin]
```

Resolution behaviour and roughness:

```
python code/variance_vs_resolution.py <obj> <density> <period>
python code/slope_vs_resolution.py    <obj> <density> <period>
python code/smooth_test.py            <obj> <density> <period>
```

Interior-density pilot (developed further in Paper V):

```
python code/density_estimate.py "Itokawa Hayabusa 50k poly.obj" \
       --xsplit 0.150 --mtotal 3.58e10 --period 12.1324
```

## 4. Definitions and conventions

Lengths are in km in the shape files and converted to SI internally.

- Attraction **A** is the self-gravitational acceleration (points into the body).
- Centrifugal acceleration is **w^2 r_perp**, directed away from the spin axis.
- Effective acceleration: **a_eff = A + w^2 r_perp**.
- Gravitational slope: theta = angle between **-a_eff** and the outward facet
  normal; zero on a surface perpendicular to the local effective gravity.
- Geopotential: V = U_grav - (1/2) w^2 r_perp^2, with U_grav negative
  (the `polyhedral-gravity` library returns the opposite sign; the code negates it).
- Normalized geopotential dispersion: area-weighted standard deviation of V
  divided by the absolute area-weighted mean of V (dimensionless). It is a
  coefficient of variation, not a variance; the paper therefore calls it a
  dispersion.

**Sign-convention warning.** The two solvers store gravity differently:
`surface_slope.py` (pure numpy) stores an OUTWARD-pointing vector, while
`surface_slope_lib.py` (library) returns the INWARD attraction. The centrifugal
term therefore enters with opposite signs in the two files. A non-rotating
sphere test CANNOT detect an error here, because it gives zero slope for either
sign. Use `verification/rotsphere_check.py`, which discriminates sharply.

## 5. Shape models used

| Body | Model | Facets | Density (g/cm^3) | Period (h) |
|---|---|---|---|---|
| (101955) Bennu | OLA v20 200k | 196,608 | 1.194 | 4.296061 |
| (162173) Ryugu | SHAPE_SFM_49k_v20180804 | 49,152 | 1.2 | 7.627 |
| (433) Eros | Gaskell 50k | 49,152 | 2.67 | 5.27 |
| (25143) Itokawa | Gaskell 50k | 49,152 | 1.95 | 12.132 |
| (25143) Itokawa | radar-based | 3,688 | 1.95 | 12.132 |
| (951) Gaspra | Thomas | 32,040 | 2.7 | 7.042 |
| (243) Ida | Thomas | 32,040 | 2.6 | 4.634 |
| (216) Kleopatra | Ostro radar | 4,092 | 3.38 | 5.385 |
| (216) Kleopatra | kleo.v2.7.1148.mod | 2,292 | 3.38 | 5.385 |

Sources: OSIRIS-REx / asteroidmission.org (Bennu); JAXA DARTS Hayabusa2 archive
for Watanabe et al. 2019 (Ryugu shape and per-facet gravity products);
PDS Small Bodies Node and JAXA DARTS (remaining models).

Ryugu reference products:
`https://data.darts.isas.jaxa.jp/pub/hayabusa2/paper/Watanabe_2019/`

## 6. Principal results (all in results/all_results.json)

**Verification.** Potential against the closed-form homogeneous ellipsoid:
Pearson r = 1.000000, mean relative error 0.18%. Slope against the analytic
rotating sphere at k = w^2R/g = 0.5: agreement within 0.14 deg.

**Ryugu, per facet, against the Hayabusa2 products** (identical shape, density,
and spin; 49,152 facets): mean slope 15.06 deg against 15.06 deg; RMS difference
0.48 deg; 49,133 of 49,152 facets (99.96%) agree within 5 deg and 49,127
(99.95%) within 2 deg; 19 facets differ by more than 5 deg and seven by more
than 10 deg, the largest by 24.45 deg; Pearson r = 0.9989 in slope and 0.9898
in geopotential.

**Bennu.** Mean slope 15.45 deg at 30,561 facets and rho = 1.194. Density sweep:
25.55 deg at rho = 0.85 and 16.06 deg at rho = 1.15, against roughly 24 and 15 deg
reported from OSIRIS-REx analyses. Equatorial mean 11.19 deg and polar 17.81 deg,
against roughly 12 and 18 deg.

**Resolution behaviour** (2,000 to 42,000 facets, five bodies): the
area-weighted mean slope varies by 1.8% (Gaspra) to 25.6% (Ryugu), the
normalized geopotential dispersion by 0.9% (Itokawa) to 5.9% (Ryugu). The
dispersion is the more stable diagnostic on every body. Between the two finest
meshes the dispersion is already near a plateau (0.00-0.60%), whereas the mean
slope shows no comparable plateau within the tested resolution range and is
still changing by up to 8.24%.

**Cross-version comparison, Itokawa** (Gaskell vs radar at ~3,688 facets):
area-weighted mean slope agrees to 0.8% and geopotential dispersion to 7.9%,
but the
tails do not — maximum slope 149 deg vs 41 deg, and 4.8% vs 1.7% of facets above
30 deg. Smoothing the Gaskell model for 10 Laplacian iterations gives 12.90 deg,
within 3.7% of the independent radar model (13.40 deg), indicating that the
difference between shape versions is dominated by facet-scale roughness.

**Kleopatra is excluded from the resolution analysis**: both available meshes
saturate under decimation (4,092 and 2,292 facets). The two Kleopatra models
also differ in scale (long axis 276 km vs about 217 km), so their cross-version
difference conflates scale, aspect ratio, and surface detail; it is reported
only as a caveat.

## 7. Known limitations

- All fields assume a homogeneous interior. This is exact for none of these
  bodies; Itokawa in particular has a denser head than body.
- Slope values are meaningful only together with the mesh resolution at which
  they were computed.
- `density_estimate.py` is a pilot: the head region is closed with a planar cap
  carrying about a 3% volume bias, and the head-only model has no interior
  minimum. Replacing the cap with a true interior volume mesh and adding a
  compressed-neck model is reserved for future work.

## 8. Numerical-correctness note

An earlier version of this code carried the centrifugal acceleration with the
wrong sign, so that it reinforced rather than opposed self-gravity, and the
rotational term of the geopotential was inconsistent with the gravitational
term. All results in this package postdate that correction, which was found and
fixed by comparison with the analytic rotating-sphere solution now provided in
`verification/rotsphere_check.py`. Superseded values are retained inside
`all_results.json` under `prev_wrongsign_*` keys for transparency.
