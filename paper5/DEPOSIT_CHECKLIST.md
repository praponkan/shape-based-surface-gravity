# Deposit checklist

Everything in this package is finished except the DOIs, which cannot be filled
in until they are reserved. Work through the steps below in order; the whole
thing takes about twenty minutes.

## 1. Reserve the DOIs

On Zenodo, create a new upload and press **Reserve DOI** before publishing.
That gives an identifier you can put into the files, so the deposit and the
manuscript can cite each other.

If the manuscript is deposited separately, reserve a second DOI for it. If the
manuscript and this archive go into one deposit, one DOI serves both.

## 2. Fill the placeholders

    python set_doi.py --deposit 10.5281/zenodo.NNNNNNN \
                      --paper   10.5281/zenodo.MMMMMMM

One DOI for both:

    python set_doi.py --deposit 10.5281/zenodo.NNNNNNN \
                      --paper   10.5281/zenodo.NNNNNNN

Then confirm none remain:

    python set_doi.py --check

This touches `.zenodo.json`, `CITATION.cff` and `manuscript/Section9.tex`.

## 3. Recompile the manuscript

Section 9 changed, so the PDF must be rebuilt. Compile twice, so that the
bibliography numbering settles:

    pdflatex main && pdflatex main

Before uploading, open the PDF and check:

- the data availability statement carries the real DOI, not the placeholder
- no `[?]` appears anywhere, which would mean a citation did not resolve
- all seven figures are present and are the current versions; Figure 7 should
  read "mesh-resolution sensitivity envelope", not "Paper IV noise band"

## 4. Verify the archive against itself

    cd data && python ../code/check_key_results.py

This reads the archived JSON and checks the values quoted in the manuscript. It
needs no shape models and no gravity library, and finishes in about a second.
Every check should pass. If one does not, the manuscript and the records have
drifted apart and that must be resolved before publishing.

## 5. Upload

Upload the whole directory, or a zip of it, plus the compiled manuscript PDF.

Name the PDF to match the rest of the series, which uses
`Surname_Year_technique_subject_PaperN.pdf`:

    Kanjanatarayont_2026_potentialdispersion_interiordensity_PaperV.pdf

The earlier papers are named the same way, so a reader who has one can guess
the others:

    Kanjanatarayont_2026_areaequivalentradius_surfacegravity_PaperI.pdf
    Kanjanatarayont_2026_optimalradiusblend_P3P2law_PaperII.pdf
    Kanjanatarayont_2026_shapebased_surfacegravity_PaperIII.pdf
    Kanjanatarayont_2026_geopotentialdispersion_surfacegravity_PaperIV.pdf

Zenodo reads `.zenodo.json` for the title, description, creators, licence and
keywords, so those fields should already be filled when the form loads. Check
that the ORCID has come through.

## 6. Publish, then record the DOI

After publishing, note the DOI in the paper's own reference list if the
manuscript cites the deposit, and in any submission to a journal.

---

## What is in the package

| Directory | Contents |
|---|---|
| `code/` | eleven scripts: the region solver, the decimator, the synthetic generator, the analytic ellipsoid test, the batch runners, the figure scripts, and the verification script |
| `data/` | the consolidated results, the master value file, the run records, mesh diagnostics, the shape-model sources, and the decimation notes |
| `figures/` | the seven figures as they appear in the paper |
| `manuscript/` | LaTeX sources: `main.tex`, Sections 1–9, Appendices A–C |

Shape models are **not** included. They are third-party products under the
licences of their archives; `data/SHAPE_MODELS.md` gives the source, the file
as used, and the enclosed volume for each, together with three properties worth
knowing before reuse — the Itokawa volume offset, the two unjoined shells of
the Arrokoth model, and the unclosed 67P SPC model.

## Reproducing the results

Figures can be regenerated from the archived JSON with `numpy` and
`matplotlib` alone:

    cd data && python ../code/make_figs.py && python ../code/make_figs2.py

Rerunning the inversions needs the shape models and a polyhedron gravity
binding, and takes about fourteen hours:

    python code/run_all_paperV.py --meshcache meshcache

The decimated meshes are cached, so a second run is much faster. Decimating
67P from 1.3 million to 50,000 facets takes about eight minutes on its own.

The analytic ellipsoid test of Section 8.3 is independent of every shape model
and takes half an hour:

    python code/test_ellipsoid.py
