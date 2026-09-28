# Shape models

The shape models are not redistributed here. All five are publicly archived by
the missions or teams that produced them, and each is available from the source
below. The table gives the file as used, so that a reader can confirm they have
the same one: a body often has several models of different provenance and
resolution, and they do not enclose identical volumes.

| Body | File as used | Native facets | Volume (km³) | Archive |
|---|---|---|---|---|
| 25143 Itokawa | `Itokawa Hayabusa 50k poly.obj` | 49,152 | 0.0177856 | PDS Small Bodies Node, Hayabusa/AMICA shape models (Gaskell et al.) |
| 433 Eros | `Eros Gaskell 50k poly.obj` | 49,152 | 2506.39 | PDS Small Bodies Node, NEAR/MSI shape models (Gaskell) |
| 101955 Bennu | `Bennu_v20_200k.obj` | 196,608 | 0.0615530 (at 50,000 facets) | OSIRIS-REx mission archive, OLA v20 |
| 67P/Churyumov–Gerasimenko | `cg_mspcd_shap2_001m_cart.obj` | 1,309,996 | 18.5912 (at 50,000 facets) | ESA Planetary Science Archive, Rosetta/OSIRIS SHAP2 |
| 67P (cross-version) | `Churyumov-Gerasimenko SPC 2017 - 199k poly.obj` | 199,000 | 18.7558 (at 49,998 facets) | ESA Planetary Science Archive, SPC 2017 |
| (486958) Arrokoth | `arrokoth_porter_2024_v01.obj` | 40,960 | 4123.81 | Porter et al. (2024), *PSJ* **5**, 260, supplementary material |

Volumes are those of the mesh actually used. Where a mesh was decimated, the
figure is the decimated volume; the native volume differs by less than 0.03%
in every case (Section 3.2).

## Points worth checking before use

**Itokawa.** The model encloses 0.0177856 km³ against the
(1.84 ± 0.092)×10⁷ m³ adopted by Abe et al. (2006), a difference of −0.67σ.
This is an ordinary difference between shape models and well inside the
published uncertainty; Section 3.8 shows the contrast and excess mass move by
under one per cent when the model is rescaled to the published volume.

**Arrokoth.** The model is supplied as **two unjoined shells**: Euler
characteristic 4 over two connected components, with no repeated or unmatched
directed edges. The lobes touch geometrically but are separate surfaces
topologically. Any code that assumes a single closed surface will need to
handle this; a region cut beyond the neck is legitimately in two pieces.

**67P SPC 2017.** This model is **not a closed surface as supplied**, carrying
one repeated and two unmatched directed edges. The defect is inherited by any
region cut from it. It does not affect a volume or a potential, both of which
are sums over oriented facets, and the cross-version agreement reported in
Section 4.4 is evidence of that; but a manifold check will fail on it.

**67P, third model.** A decimated DLR product of 200,000 facets was also
examined and is not used: it encloses 3.9% less volume than the published
figure, against 0.07% and 0.22% for the two models above.

## Decimation

Meshes above 50,000 facets were reduced by quadric edge collapse with the link
condition enforced (`decimate_qem.py`), which preserves the manifold property
and returns the requested facet count exactly. Papers III and IV used vertex
clustering, which is equally accurate in the fields it produces but does not
preserve topology; see `DECIMATION_NOTES.txt` and Section 8.3.

To reproduce a run exactly, the decimated meshes must match. `density_estimate.py`
caches them under `--meshcache <dir>` as `.npz` files keyed by shape model,
method and target, so that every run of the same body uses the identical mesh.
Decimating 67P from 1.3 million to 50,000 facets takes about eight minutes; the
cache pays for itself immediately.
