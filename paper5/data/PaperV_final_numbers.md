# Paper V — final numbers

Every figure that may appear in the manuscript, in one place, with its
provenance. Several values were corrected during the work, in some cases more
than once; where that happened the superseded value is shown struck through so
that an old version cannot be picked up by mistake.

**Rule.** If a number is needed and it is not in this file, do not write it
from memory. Go back to `all_results.json` or to the run records.

---

## 0. ⚠️ Where this file contradicts itself

Sections written before the rerun of 2026-08-19 carry 62,876-facet Bennu values
and, in one place, the wrong sign on the centre-of-mass offset. **Section 12 is
authoritative.** The Bennu values for the paper are:

| quantity | value |
|---|---|
| facets | 50,000 |
| ρ_R / ρ_rest | 1190 / 1192 |
| contrast | 0.999 (0.998578) |
| improvement | 0.000% (0.000420%) |
| ΔM | −0.0715% |
| **COM offset** | **−0.07 m** — negative, as ΔM is |
| σ(ρ) | ±42 → contrast 1.000 ± 0.035 |

**Two further values were corrected in the manuscript on 2026-08-22**, after
`check_key_results.py` compared it against the archived runs. Both bodies are
undecimated, so the change comes from the per-edge intersection cache in the
clipper, not from the decimator; both are below one part in a thousand and
neither affects a conclusion.

| where | was | **is** |
|---|---|---|
| Itokawa excess mass (§3.1, §5.4) | 2.7527×10⁹ kg | **2.7525×10⁹ kg** |
| Eros improvement (§4 table, §4.1) | 0.417% | **0.418%** |

These two are no longer checked by `check_key_results.py`; it verifies 62
values and passes all of them.

## 1. Bodies and configurations

| body | shape model | facets used | volume (km³) | mass (kg) | bulk (kg m⁻³) | period (h) |
|---|---|---|---|---|---|---|
| Itokawa | Itokawa Hayabusa 50k poly.obj | 49,152 | 0.0177856 | 3.58×10¹⁰ | 2012.9 | 12.1324 |
| Eros | Eros Gaskell 50k poly.obj | 49,152 | 2506.39 | 6.688×10¹⁵ | 2668.4 | 5.27 |
| Bennu | Bennu_v20_200k.obj | 62,876 | 0.0615364 | 7.330×10¹⁰ | 1191.2 | 4.296061 |
| 67P | cg_mspcd_shap2_001m_cart.obj | 57,607 | 18.5475 | 9.982×10¹² | 538.2 | 12.4043 |
| Arrokoth | arrokoth_porter_2024_v01.obj | 40,960 | 4123.81 | assumed | assumed | 15.92 |

**Masses used.** Itokawa 3.58×10¹⁰ kg. Eros: 6.688×10¹⁵ computed as
2670 × model volume; the measured value is 6.687×10¹⁵ (Yeomans et al. 2000),
differing by 0.015% — quote the measured one. 67P: 9.982×10¹² (Pätzold et al.
2016). Bennu: 7.330×10¹⁰ from GM = 4.892×10⁻⁹ km³ s⁻². Arrokoth has **no
measured mass**; three assumptions were run, 250, 400 and 500 kg m⁻³, giving
1.0310×10¹⁵, 1.6495×10¹⁵ and 2.0619×10¹⁵ kg.

**The Itokawa volume offset is NOT significant.** Abe et al. (2006) give the
volume as (1.84 ± 0.092)×10⁷ m³, a 5% uncertainty. Our model's 0.0177856 km³
is **−0.67σ** from it, and our bulk density of 2.013 g/cm³ is **+0.45σ** from
the published 1.95 ± 0.14 (7%). Stop describing this as a systematic that
inflates the densities; it is an ordinary difference between shape models,
inside the published error bar. The scale test still matters, because it shows
contrast and ΔM move by under 1% when the volume is changed.

**Volume offsets against published values.**

| body | model | published | offset |
|---|---|---|---|
| Itokawa | 0.0177856 km³ | 0.01840 km³ | **−3.34%** |
| Eros | 2506.39 km³ | ~2503 km³ | +0.14% |
| Bennu | 0.0615364 km³ | — | bulk 1191.2 vs 1194, +0.2% |
| 67P MSPCD | 18.5475 km³ | 18.56 km³ (Preusker 2017) | −0.07% |
| 67P SPC | 18.7404 km³ | — | bulk 532.6 vs 533 (Pätzold), −0.08% |
| 67P DLR 200k | 17.8396 km³ | 18.56 km³ | −3.88%, **not used for science** |
| Arrokoth | 4123.81 km³ | 4123.78 km³ (Porter 2024 Table 2) | +0.001% |

---

## 2. Itokawa — the one detection

**Reference run**, unreduced mesh, plane at the detected neck x = 0.1608 km:

| quantity | value |
|---|---|
| ρ_head | 2800 kg m⁻³ |
| ρ_rest | 1858 kg m⁻³ |
| contrast | 1.507 |
| dispersion | 0.072000 |
| improvement over uniform | +16.889% |
| region volume fraction | 16.4% |
| region centroid x | +0.2197 km |
| excess mass ΔM | 2.7525×10⁹ kg |
| COM offset | 16.86 m |
| σ(ρ) at 0.60% noise | ±130 kg m⁻³ |

**Neck model at the same plane:** ρ_neck 2525, ρ_rest 1943, contrast 1.300,
improvement +2.275%, σ ±261 (10.3% of the value). No detection.

**Split-plane sweep**, 12 planes, 0.130 to 0.240 km. Slope 7.5 kg m⁻³ per
metre over 0.130–0.160 with zero residual, then 10.0, 12.5, 20.0, 27.5, 55.0.

Over the physically allowed range (see §5): ρ_head ±13.1%, ΔM ±5.1%,
contrast ±13.5%, COM offset ±12.8%. Scaling slope **−1.14**.

**Mesh convergence.** Two fixed-plane pairs, both giving exactly zero change:
21,196 → 34,234 facets at x = 0.1702 (+61% facets); 41,369 → 49,152 at
x = 0.1608 (+19%).

---

## 3. The four nulls

| body | plane | region % | contrast | improvement | note |
|---|---|---|---|---|---|
| Eros | +4.4698 km (Himeros) | 34.6 | 1.018 | +0.418% | σ ±51 → contrast 1.000 ± 0.019 |
| Bennu | x = 0 (imposed) | 50.2 | 0.998 | +0.001% | COM offset 0.10 m |
| 67P | +0.9470 km (neck) | 26.68 | 1.069 | +0.606% | |
| Arrokoth | −4.7863 km (neck) | 66.33 | 0.964 | +0.883% | at ρ = 400 |

**Eros at the misplaced plane** (x = −10.7619 km, the detector's original
choice): contrast 0.875, improvement +5.539%, COM offset −216.5 m. Thirteen
times the correct-plane signal, four times the COM offset, opposite sign.

**67P cross-version.** MSPCD vs SPC at the same plane: ρ_R 565 in both,
contrast 1.069 vs 1.084, improvement 0.606% vs 0.698%.

**Arrokoth mass sensitivity**, at the neck:

| assumed ρ | contrast | ΔM/M | COM offset |
|---|---|---|---|
| 250 | 1.062 | +3.93% | +226 m |
| 400 | 0.964 | −2.46% | −141 m |
| 500 | 0.916 | −5.91% | −340 m |

Sign reversal at **≈339 kg m⁻³**, just above the ~290 lower bound of Spencer
et al. (2020).

**Arrokoth fraction sweep** (7 planes, 40.0% to 75.6%, V_R span 1.890): Δρ
runs from +33.3 to −61.5 kg m⁻³ and **changes sign**; at a region fraction of
61.2% the inversion returns the uniform solution exactly (400/400, contrast
1.0000, improvement −0.000%). No scaling slope exists. Improvement range
−0.000% to 5.352%.

**Arrokoth mesh convergence**, plane fixed at the neck: ρ_R 400 → 395 → 395
for 19,399 → 29,194 → 40,960 facets. Zero change between the two finest; one
scan step (5 kg m⁻³, 1.25%) coarsest to finest, falling to 0.88% once the
coarse mesh's 0.375% volume loss is allowed for.

---

## 4. Residual dispersion after the best fit

| body | uniform | best fit | improvement | residual |
|---|---|---|---|---|
| Itokawa | 0.0866 | 0.0720 | 16.90% | **7.20%** |
| 67P | 0.0758 | 0.0754 | 0.61% | 7.54% |
| Bennu | 0.0616 | 0.0616 | 0.00% | 6.16% |
| Arrokoth | 0.0601 | 0.0595 | 0.88% | 5.95% |
| Eros | 0.0341 | 0.0340 | 0.42% | **3.40%** |

---

## 5. Density ceilings — report both, they mean different things

| value | name | meaning | bound on x_split |
|---|---|---|---|
| 3540 ± 130 | LL **grain** density | absolute: nothing exceeds the mineral matrix | x ≤ 0.2069 km (range 0.2022–0.2116) |
| 3220 ± 220 | LL **meteorite bulk** density | realistic: a pile of chondritic blocks cannot exceed the blocks | x ≤ 0.1935 km |
| 3190 | the value used throughout the Itokawa literature | same as the row above | x ≤ 0.1920 km |

**Settled from the primary sources.** Abe et al. (2006): "the **bulk density**
of LL ordinary chondrites of 3.19 g/cm³". Fujiwara et al. (2006): "a typical
value for their **bulk density** is 3.2 g cm⁻³". Kanamaru calls the same
number a *grain* density and cites Abe for it — a misattribution that has
propagated. Consolmagno et al. (2008) give LL bulk 3.22 ± 0.22 and LL grain
3.54 ± 0.13, so the measurements agree and only the label differs. Report both
bounds and note the mislabelling in passing.

Source: Consolmagno et al. (2008), *Chemie der Erde* 68, 1–29. The upper edge
of the neck slab is at 0.2029 km, so the **realistic** bound falls inside the
neck region and the **absolute** bound 4 m beyond it.

> ~~3200 as "the grain density"~~ — wrong label. It is the bulk density.
> ~~The grain density 3540 is the value to use~~ — an over-correction; both
> ceilings are valid and mean different things.

---

## 6. Comparison at the matched plane x = 150 m

**Four independent determinations at the same dividing plane.** This is the
table for the manuscript; a single pairwise comparison wastes the material.

| study | method | ρ_head | ρ_body | contrast | COM offset |
|---|---|---|---|---|---|
| Lowry et al. (2014) | YORP spin-up, no relaxation premise | 2850 ± 500 | 1750 ± 110 | 1.629 | 21 ± 12 m |
| Kanamaru & Sasaki (2019), TJSASS | global potential variance | 2730 | 1870 | 1.460 | 16 m |
| Kanamaru et al. (2019), PSS | smooth-terrain equipotential fit | 2450 | 1930 | 1.269 | 9.9 m |
| **this work** | global dispersion | **2725** | **1858** | **1.467** | **16.5 m** |

This work agrees almost exactly with the study that shares its method. All
four contrasts lie inside Lowry's uncertainty, which spans roughly 1.26–2.04
once ±500 and ±110 are propagated. Lowry's bulk is held at 1950; ours is 2013
(model volume) — see §7.

⚠️ Lowry also cut the body at x = 150 m. That the three methods converge on
the same plane is a coincidence of convention worth noting, not evidence.

## 6b. Detail on Kanamaru & Sasaki (2019)

Trans. JSASS Aerospace Tech. Japan **17**(3), 270–275, doi:10.2322/tastj.17.270

| | ρ_head | ρ_body | contrast | COM offset |
|---|---|---|---|---|
| Kanamaru | 2730 (2750 in the abstract) | 1870 | 1.460 | 16 m |
| this work at x = 0.150 km | 2725 | 1858 | 1.467 | 16.5 m |
| difference | −0.2% | −0.6% | **+0.46%** | **+3.1%** |

Kanamaru's split plane: X = 150 m. Itokawa bulk 1950 ± 140. Macroporosity 14%
for the head, 41% for the body, against LL mean 3190.

> ~~ρ_body = 1930, contrast 1.425, agreement +2.9%~~ — 1930 belongs to the
> **other** Kanamaru paper (Kanamaru, Sasaki & Wieczorek 2019, *Planet. Space
> Sci.* **174**, 32–42), which uses smooth terrain and a different method.

---

## 7. The volume-offset scale test

Mesh enlarged by a linear factor 1.011385 to the published volume; the plane
moved with it from 0.1608 to 0.1627 km.

| quantity | model scale | published scale | change |
|---|---|---|---|
| contrast | 1.50692 | 1.50215 | **−0.32%** |
| ΔM | 2.7525×10⁹ | 2.7273×10⁹ | **−0.92%** |
| COM offset | 16.863 m | 16.900 m | +0.22% |
| improvement | 16.888% | 16.883% | −0.03% |
| ρ_R | 2800 | 2700 | −3.57% |
| ρ_rest | 1858.1 | 1797.4 | −3.27% |

Centrifugal over gravitational **force** at the equator: 3.678% → 3.805%.

**Reporting decision.** Report all three: contrast and ΔM as the primary
results with a note that they are insensitive to the offset; absolute
densities as computed, always with the volume and bulk density they rest on.
Do not quote a naively rescaled density.

> ~~ρ_head rescaled = 2708, ρ_rest = 1797~~ — withdrawn as a *method*. The
> measurement gives 2700 and 1797, within one scan step of the naive
> prediction, but the agreement is a property of Itokawa's slow rotation and
> is not a general licence.

---

## 8. Richardson & Bowling (2014) — verified from the paper

*Icarus* **234**, 53–65, doi:10.1016/j.icarus.2014.02.015

The four conditions, verbatim: (1) the mean rotational force is a significant
fraction of the mean gravitational force; (2) a sufficiently thick, low
cohesion, mobile regolith layer exists over most of the body's surface; (3) a
downslope flow disturbance source is present; (4) a sufficient amount of time
has occurred since the body's last major surface alteration.

Their Table 1, mean rotational / mean gravitational **potential**:

| body | ratio | global estimate | regional | measured |
|---|---|---|---|---|
| 103P Hartley 2 | 0.134 | 200 | 220 | — |
| 243 Ida | 0.131 | 2300 | 2900 | 2600 ± 500 |
| 433 Eros | 0.111 | 2200 | 2300 | 2670 ± 30 |
| 951 Gaspra | 0.054 | 900 | 1400 | — |
| Phobos | 0.039 | 2200 | 2300 | 1876 ± 10 |
| **25143 Itokawa** | **0.024** | **330** | 2000 | **1950 ± 140** |
| 9P Tempel 1 | 0.006 | 28 | 130 | 400 ± 200 |

They state that Itokawa's global result is "indicative of a body on which
rotational forces do not generally affect slope directions to a high enough
degree to be significant (the first criterion)". Itokawa is in their Rotation
Group II.

⚠️ Their 0.024 is a ratio of mean **potentials**. The 3.678% in §7 is a ratio
of **forces** at the equator. Different quantities — never present them as the
same number.

---

## 9. Synthetic invariance experiment

Eight bodies, volumes within 7%, V_R span 1.90 on every one.

| regime | bodies | improvement | slope |
|---|---|---|---|
| no anomaly | sphere, sym_waist, sym_neck, sym_deep | ≤4.5% | none; Δρ changes sign |
| moderate | asym_waist, asym_neck, asym_deep | 16.7–19.5% | −2.02, −2.37, −2.07 |
| strong | asym_extreme | 43.7% | −1.22 |

Correlation between signal strength and slope: **0.95**. asym_extreme: ρ
±12.78%, ΔM ±6.22%, against Itokawa's ±13.1% and ±5.1%.

Sliver control: rerun at `--cliptol 1e-6`, every printed digit identical.

---

## 10. Numbers that must NOT appear

| forbidden | why |
|---|---|
| "Lowry gives only a COM offset" | they also give 2850 ± 500 / 1750 ± 110 at x = 150 m |
| recomputing Richardson's 0.024 from ω²/(4πGρ) | that sphere formula gives ≈0.013; their table uses the real shape |
| "Bennu is homogeneous" | Scheeres et al. (2020) report an under-dense equator and centre; our null concerns left–right asymmetry only |
| grain density 3200 | it is the bulk density; use 3540 or 3220 with the right label |
| Kanamaru ρ_body 1930, contrast 1.425 | belongs to the other Kanamaru paper |
| ρ_head "rescaled" 2708 | withdrawn as a method; use the measured 2700 |
| Itokawa split-plane slope 10.6 kg m⁻³ per metre | two-point estimate, superseded by the 12-point sweep |
| Arrokoth scaling slope from the position-based sweep | V_R span was 1.05; superseded by the fraction-based sweep |
| a 15–85% validity window | proposed, tested and withdrawn |
| any threshold signal strength quoted as universal | the synthetic threshold does not transfer |
| "Itokawa fails the rotation criterion because 3.68% is small" | 3.68% is a force ratio; Richardson & Bowling's criterion is stated on potentials, where Itokawa is 0.024 |

---

## 10b. Mass-sensitivity test (five runs, identical mesh)

| bulk (kg m⁻³) | uniform disp | ρ_R | ρ_rest | contrast | improvement |
|---|---|---|---|---|---|
| 1006 | 0.069569 | 1325 | 943.8 | 1.4039 | 16.298% |
| 1510 | 0.080812 | 2075 | 1398.5 | 1.4837 | 16.736% |
| **2013** | **0.086632** | **2800** | **1858.1** | **1.5069** | **16.888%** |
| 3019 | 0.092576 | 4275 | 2772.4 | 1.5420 | 17.004% |
| 4026 | 0.095595 | 5775 | 3681.8 | 1.5685 | 17.052% |

**Headline:** a fourfold change in assumed mass moves the contrast by only
**11.0%** and never near unity. Compare Arrokoth, where a factor of two
reversed the sign. Sensitivity to the assumed mass is itself a signal
diagnostic.

> ~~The dispersion is independent of assumed mass (a plateau)~~ — **retracted.**
> It rises 37.4% over the factor of four, log-log slope +0.224. The ρ→∞ limit
> is real but Itokawa is not in it: the rotational potential is 2.4% of the
> *mean* yet a large part of the *spread*, because it varies as the square of
> the distance from the spin axis while the gravitational potential is
> comparatively uniform.

## 11. Cross-check against Richardson & Bowling's Fig. 8

Their y axis, "Normalized Potential Standard Deviation", is the same quantity
as our σ̃_U. For Itokawa the **global** curve plateaus at about **0.1** above
roughly 1000 kg m⁻³; our uniform value is **0.0866**. Consistent within the
precision of reading a logarithmic figure, which is not better than ~15%.
Describe it as a consistency check, not as precise agreement.

**The right comparison.** Their plateau is the ρ→∞ asymptote. Extrapolating
our five mass-sensitivity points linearly in 1/ρ gives an asymptote of
**0.1041**, against their **≈0.1**. Comparing our 0.0866 at the true density
against their plateau would have been comparing two different points on the
same curve.

**Why our inference survives where theirs fails** — the honest version, stated
empirically rather than argued: the contrast is measured to move only 11% over
a fourfold change in assumed mass, while the bulk-density inference on the same
body lands at 330 against a measured 1950. The two problems move in different
directions in parameter space. Do **not** claim the dispersion is independent
of the assumed mass; it is not.

## 12. Values superseded by the full rerun of 2026-08-19

The rerun used the corrected clipper and quadric decimation. Undecimated
results are unchanged to every digit; only decimated bodies moved.

| quantity | old | **new** |
|---|---|---|
| 67P facets | 57,607 | **50,000** |
| 67P volume | 18.5475 km³ | **18.5912** |
| 67P bulk density | 538.2 | **536.9** |
| 67P neck position | 0.9470 km | **0.9517** |
| 67P prominence | 0.122 | **0.091** |
| 67P contrast | 1.069 | **1.073** |
| 67P improvement | 0.606% | **0.562%** |
| 67P ΔM/M | 3.429% | **1.904%** |
| Bennu facets | 62,876 | **50,000** |
| Bennu volume | 0.0615364 | **0.061553** |
| Bennu contrast | 0.998 | **0.9986** |
| Itokawa mesh series | 21,196 / 34,234 / 41,369 | **20,000 / 30,000 / 40,000** |

**Unchanged:** Itokawa reference and all twelve sweep planes, Eros at both
planes, Arrokoth at every plane and mass, the entire synthetic family, and the
comparison with Kanamaru and Lowry at x = 150 m.

**Decimator sensitivity**, measured directly: contrast moves by 0.005% at a
16% reduction (Itokawa), 0.054% at 68% (Bennu), and −3.9% at 96% (67P).

## 13. New facts to report

- The Arrokoth model of Porter et al. (2024) is **two unjoined shells**
  (χ = 4, two components). A region cut beyond the neck is legitimately in two
  pieces; the manifold criterion compares χ against 2× the component count.
- The 67P SPC 2017 model is **not closed as supplied** (1 repeated, 2
  unmatched directed edges).
- Vertex clustering does not preserve topology, and the damage grows with the
  reduction: 96 repeated edges at 43%, 1,449 at 96%.
- Vertex clustering **ignores the requested target below a certain size**:
  1200, 800, 500 and 300 all return the same 2,159-face mesh.

## 14. The ellipsoid test — an analytic reference

A homogeneous ellipsoid has exact surface gravity (MacMillan): the sphere
limit reproduces GM/R² to 2.2×10⁻¹⁴%. Three ellipsoids × three spin states ×
five facet counts, judged against quadrature on the exact surface.

**Mean absolute error against truth:**

| | slope | dispersion |
|---|---|---|
| QEM | 4.05% | 0.09% |
| clustering | **3.92%** | **0.07%** |

**The two decimators are equivalent in accuracy.** Clustering is marginally
better on a smooth surface.

**Dispersion against slope, the Paper IV question:**

| faces | slope err | dispersion err | ratio |
|---|---|---|---|
| 2,000 | 11.62% | 0.196% | **59×** |
| 8,000 | 2.30% | 0.049% | **47×** |
| 32,000 | 0.33% | 0.017% | **19×** |

**Paper IV is confirmed and understated.** It inferred a ratio of 3–14 from
mesh-to-mesh variation; against an analytic reference it is 19–59.

> ~~QEM is more accurate than clustering~~ — **retracted.** They are
> equivalent. QEM is kept for topology (manifold preservation, exact facet
> count), not accuracy.
>
> ~~Clustering biases slope downward by merging ridges~~ — **retracted.**
> Both methods *overestimate* slope on coarse meshes, by the same amount, in
> all 90 configurations, and converge smoothly. The monotonic drift seen on
> real bodies is not reproduced on a smooth ellipsoid and remains unexplained;
> roughness is a hypothesis, not a measurement.
>
> ~~Paper IV needs a corrected version~~ — **retracted.** Its conclusion
> holds and is conservative.

## 15. Settled since this file was started

1. **Paper IV needs no correction.** The ellipsoid test of §14 confirms its
   conclusion and shows it to be conservative. A supplement is prepared —
   `surface_slope_lib.py` with a decimator dispatcher added and
   `decimate_cluster` untouched, `decimate_qem.py`, `DECIMATION_NOTES.txt` —
   to be issued as a new version of the existing deposit, not as a correction.
2. **The plateau was measured, not inferred.** Five runs varying only the
   assumed mass (§10b) show the dispersion rises 37.4% across a factor of
   four, so Itokawa is not on the plateau; the earlier plateau argument is
   retracted. What the test established instead is that the contrast moves by
   only 11% over the same range.

## 16. Still open

**Paper III checked and clean** — it never invokes relaxation or potential
variance, and states that GM must be known independently. Note that the
"Richardson" it cites is Derek C. Richardson on rubble-pile shapes, a
different author from James E. Richardson.
