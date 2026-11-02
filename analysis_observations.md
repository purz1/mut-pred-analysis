# Analysis observations and caveats

Observations made while redesigning the cytomorphology / attention figures (Oct 2026).
Things to keep in mind when interpreting results and writing the manuscript.
Data version: `03-06-2026`.

---

## 1. Vacuolation is linked to cell size, independent of genetics

- Within each patient, vacuolated cells are larger than non-vacuolated cells of the same type
  (larger cell and nucleus area, lower N:C ratio), in every cell type tested.
  Median within-patient Cliff's δ for cell area: ~0.3–0.4 in blasts / myeloid cells, ~0.6–0.7 in lymphocytes.
- The size effect is about the same in every patient group (all 17 target variables),
  so it is **not a mutation-specific signature**.
- Likely explanations: (a) the vacuole model flags larger cells more easily (detection bias), or
  (b) pre-analytical swelling + vacuolation (e.g. EDTA storage time before smearing).
- **Implication:** interpret vacuolation associations (e.g. TP53 / complex karyotype) cautiously.
- **To do:** check whether the proportion of vacuolated cells varies with sample-level factors
  (cohort, collection year / `database_id`) as a proxy for sample handling.

Outputs: `data/03-06-2026/derived/morph_dysplasia/vacuolation_*.csv`

## 2. Dysplasia classes are reflected in morphometry

Cell-level, within-patient comparison of dysplastic vs non-dysplastic cells of the same type:

- **Bilobed nucleus** (blasts, promyelocytes): lower nucleus solidity and compactness.
- **Hyposegmented neutrophils** (PB): higher solidity / compactness, smaller nucleus.
- **Hypergranular blasts / myelocytes:** smaller cells, higher N:C.
- **Erythroblasts dysmorphic / multinucleated:** strong shape changes, but only 10–12 patients.
- **Vacuolated:** size changes only, little nuclear shape change (see section 1).
- Not testable (< 10 patients with enough dysplastic cells): Auer rods in blasts, megakaryocyte classes, PB myelocytes.

**Implication:** some dysplasia and morphometry findings describe the same morphology
(e.g. PML-RARA: "bilobed nucleus" and "promyelocyte nucleus solidity / compactness").
Do not present them as independent findings.

Script: `scripts/extract_morphometry_dysplasia_correlation.py` →
`data/03-06-2026/derived/morph_dysplasia/morph_dysplasia_effects.csv`

## 3. Overlapping and nested target labels

- **Co-occurring labels:** TP53, complex karyotype, monosomal karyotype and −5/del(5q) give very similar
  morphology patterns. KIT (n = 15) mirrors the CBF fusion findings (KIT co-occurs with CBF AML).
- **Nested labels:** NPM1 ⊃ NPM1 + FLT3-ITD, IDH mutations ⊃ IDH2, FLT3 mutations ⊃ FLT3-ITD.
  A child label largely repeats its parent's fingerprint.
- **Implication:** a shared pattern may be driven by one variable. Consider adjusted models
  (e.g. `feature ~ TP53 + CK + MK + del5q`) and subgroup-vs-parent comparisons
  (e.g. NPM1 + FLT3-ITD vs NPM1-only) before claiming variable-specific morphology.

## 4. Morphometry of rare cell types is noisy

- Morphometry medians of rare cell types (basophils, metamyelocytes, immature eosinophils,
  megakaryocytes) give large but unstable effect sizes, often based on few samples.
- Morphometry per sample requires ≥ 20 cells (BM; megakaryocytes ≥ 10) or ≥ 10 cells (PB)
  (`core/cell_feature_aggregation/aggregate.py`); features are tested only if ≥ 10 samples per group.
- **Implication:** rank findings by FDR-adjusted significance rather than effect size alone;
  treat rare-cell-type morphometry with caution.

## 5. Attention analysis

- Attention is analysed **per modality**: BM model for BM features, PB model for PB features
  (BM+PB performance is close to BM alone).
- **PB model cell classes:** band and segmented neutrophils are merged into `Neutrophils`, and platelets
  are not modelled. Morphology data keeps band / segmented neutrophils separate.
- **Top-attention cells:** cells carrying 50% of the attention mass in each positive sample
  (`core/attention.select_top_attention_cell_ids`).
- **"Not evaluable" model focus:** too few top-attention cells of a cell type (e.g. small groups such as
  KIT / CBFB-MYH11, or cell types the model ignores). Many morphology features fall in this category,
  so absence of model focus is often "not testable" rather than "not used".
- **Mismatches between statistics and attention** are informative, e.g.:
  - PML-RARA: blasts are less frequent in positives but attended.
  - IDH: eosinophils are less frequent but strongly attended.
  - CBFB-MYH11 (BM): eosinophils increase, but the model attends monocytes / promonocytes.

Script: `scripts/extract_fig4_fingerprint_results.py` → `data/03-06-2026/derived/fig4_fingerprints/`

## 6. Cohort notes

- Discovery BM and PB cohorts both have n = 562 samples (paired patients).
- External validation is possible for ~8 BM and 4 PB variables.

---

## Open checks

- [ ] Vacuolation vs sample handling (`database_id` / cohort).
- [ ] Adjusted analysis for co-occurring labels (TP53 / CK / MK / −5q; KIT vs CBF).
- [ ] TP vs FN (and FP vs TN) differences in cell proportions, for the supplementary variable cards.
