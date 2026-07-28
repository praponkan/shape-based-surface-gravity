# Shape-based estimation of mean surface gravity for small bodies

Data and analysis code for **Paper III** of a three-paper series on estimating the
mean surface gravity ⟨g⟩ of small solar-system bodies (asteroids, small moons)
**from shape information alone** — without needing a measured surface gravity.

**Author:** Prapon Kanjanatarayont (Independent Researcher, Thailand)
[ORCID 0009-0009-4081-7422](https://orcid.org/0009-0009-4081-7422)

**Paper (Zenodo, open access):** https://doi.org/10.5281/zenodo.21622962

Companion papers:
[Paper I](https://doi.org/10.5281/zenodo.21327611) ·
[Paper II](https://doi.org/10.5281/zenodo.21328178)

---

## What this does

The mean surface gravity of a small body is almost never measured directly.
What is known is its shape (from radar or spacecraft imaging) and often its
mass parameter *GM*. This work estimates ⟨g⟩ from the shape using a **blended
radius**

```
R_w = w · R_A + (1 − w) · R_V
```

where `R_A` is the area-equivalent radius, `R_V` the volume-equivalent radius,
and `w` an **analytic** weight from the body's deformation invariants
`P2`, `P3` (no fitting). The reference ⟨g⟩ is the exact Werner–Scheeres
polyhedron field under uniform density.

### Key result

On 22 real shape models (20 distinct bodies) with deformation `P2 ≥ 0.10`:

| estimator | mean \|error\| in ⟨g⟩ |
|---|---|
| `R_V` alone | 11.54% |
| `R_A` alone | 3.93% |
| fitted ellipsoid | 2.93% |
| **w-blend (this work)** | **1.72%**  (worst 3.66%) |

The blend is the most accurate and has the smallest maximum error. The paired
advantage over the fitted ellipsoid is not significant at *n* = 22
(Wilcoxon *p* = 0.106), but a body-clustered bootstrap places the mean
advantage at **+1.18 percentage points** (95% CI +0.40 to +2.13).

---

## Quick start

Requirements: Python 3.9+, with `numpy scipy matplotlib`
(and `polyhedral-gravity` only for the reference solver).

```bash
pip install numpy scipy matplotlib
```

**Regenerate every figure from the shipped data:**

```bash
python make_paper3_figures.py
# writes paper3_fig1_accuracy.png ... paper3_fig5_spin.png
```

**Estimate ⟨g⟩ of your own polyhedral mesh, given its GM (km³/s²):**

```bash
python gravity_ws.py "your_asteroid.obj" 4.463e-4
# prints exact <g> (polyhedron), GM/R_A^2, GM/R_V^2, and each error
```

**Get the deformation invariants P2, P3 for your shape models:**

```bash
python compute_invariants.py --dir path/to/your/shapes
# writes invariants_results.json (Psi, P2, P3 per model)
```

---

## Using the blend on your own asteroid (no solver needed)

If you have the shape (or just `R_A` and `R_V`) and want ⟨g⟩:

1. Get `R_A`, `R_V` and the invariants `P2`, `P3` of the shape.
2. If `P2 < 0.10` the body is near-spherical — just use `R_V`
   (the blend weight is ill-conditioned below this gate).
3. Form the weight (Eq. 9 of the paper, **no fitting**):

   ```
   w = 19/25 + (188/875)·(P3/P2) + (722/21875)·P2 + (376/3675)·(P3/P2)²
   ```

4. Blend:  `R_w = w·R_A + (1 − w)·R_V`
5. Estimate:  `⟨g⟩ ≈ GM / R_w²`

Expect ~1.7% mean error, with a ~2 percentage-point floor from surface
roughness on real bodies. The estimate assumes uniform interior density.

---

## Files

**Code**
- `gravity_ws.py` — reference solver: exact Werner–Scheeres / Tsoulis
  polyhedron ⟨g⟩, plus `GM/R_A²` and `GM/R_V²`. Self-tests on a sphere.
- `compute_invariants.py` — deformation invariants `P2`, `P3` and axis ratios.
- `verify_numeric.py` — independent check of the analytic weight-law coefficients.
- `make_paper3_figures.py` — regenerates all five figures.

**Data**
- `convex_bias_results.json` — master per-mesh results (42 meshes; 22 in-gate at `P2 ≥ 0.10`).
- `four_method_v2.json` — four-method comparison for the 22 real models (drives Figs 1–2, matches Table A1).
- `four_method_results.json` — same comparison for 88 synthetic shapes.
- `law14_results.json` / `law14_signed.json` / `law14_perbody.json` — per-body blend errors and weights.
- `fixedgm_bias.json` — fixed-GM convex-hull bias (Result 2).
- `bilobed_results.json` — bilobedness index (Result 3).
- `spin_data.csv` — rotation period and taxonomy (Result 3).

---

## Scope and limitations

- The reference ⟨g⟩ is **computed, not measured** — no asteroid surface gravity
  has been measured directly. The polyhedron field is exact for the adopted
  uniform-density mesh.
- The blend applies for `P2 ≥ 0.10`; below that the optimal weight is
  numerically ill-conditioned.
- Convex lightcurve models understate deformation and are analysed separately
  (Result 2); they are **not** used as blend-accuracy evidence.

---

## Data provenance

Shape models are from PDS, DAMIT, and published radar and spacecraft
reconstructions; *GM* values are traced to primary mission literature.
See the paper's References and Table A1 for per-body citations.

---

## License

- **Data** (`.json`, `.csv`, figures): Creative Commons Attribution 4.0 (CC BY 4.0)
- **Code** (`.py`): MIT License

See `LICENSE.txt`.

---

## Citation

```
Kanjanatarayont, P. (2026). Shape-based estimation of mean surface gravity
for small bodies: the accuracy of the blended radius and the limits of
convex models. Zenodo. https://doi.org/10.5281/zenodo.21622962
```
