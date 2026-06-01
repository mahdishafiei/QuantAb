# Results — affinity regression across the three models

Benchmark of the three antibody language models on antibody–antigen binding-affinity
prediction. Embeddings were produced on AWS GPUs (mean-pooled, see Methods); the regression
("classification") was run by `scripts/run_regression.py`.

## Dataset
`curated_data/unified_affinity_v2.csv` — **40,937 rows, 1,041 antigens**. Each row: antibody
(heavy + light), antigen sequence, and `affinity_norm` (the regression target = `−log10 KD`
z-scored within each protocol/antigen group). Composition: Phillips CR9114 DMS (33,345),
Dailey (4,196), ANDD literature/SPR (3,310), MAGMA-seq SARS-CoV-2 (86).

## Dashboard (Spearman ρ, MLP head)

| Model (mean-pool) | Random CV — antibody | Random — +antigen | **Grouped CV — antibody** | Grouped — +antigen |
|---|---|---|---|---|
| **IgBERT**      | 0.899 ± 0.003 | 0.880 ± 0.011 | **0.482 ± 0.130** | 0.442 ± 0.139 |
| **ESM-2 650M**  | **0.907 ± 0.002** | 0.908 ± 0.009 | 0.445 ± 0.182 | 0.446 ± 0.193 |
| **BALM-paired** | 0.885 ± 0.005 | 0.868 ± 0.010 | 0.420 ± 0.106 | 0.375 ± 0.227 |

- **Random CV** = 5-fold, shuffled. Optimistic.
- **Grouped CV** = 5-fold **GroupKFold holding out whole antigens** — the honest generalization estimate.

## What the numbers say
1. **The honest number is ~0.42–0.48, not ~0.9.** Random CV is inflated by sequence-similarity
   leakage: ~80% of rows are CR9114 single-mutants, so shuffled folds put near-identical
   sequences in train and test. Holding out whole antigens (grouped) removes that, and the
   score drops sharply — this is the result to report.
2. **Pooling matters.** With mean pooling (not CLS), IgBERT leads on the honest split (0.482).
   On random CV ESM-2 leads (0.907) — but that regime is the leaky one.
3. **The antigen embedding doesn't help** (sometimes hurts). The data is flu-dominated, so the
   antigen is near-constant and adds no signal; concatenating it mostly adds noise. A more
   antigen-diverse training set would be needed to test the two-modality idea properly.
4. **High variance on the grouped split** (±0.1–0.2) reflects how different held-out antigens
   are — generalizing to a new antigen family is genuinely hard with this data mix.

## Methods (correctness)
- **Mean pooling** over residue positions (excludes special tokens) for all three models —
  CLS rejected (MEMORY.md ISSUE 2; IgBERT's pooler is untrained).
- **No leakage:** scaling/PCA fit inside each CV fold only (MEMORY.md ISSUE 1). Classical arm
  uses raw embeddings (no PCA needed).
- **Target:** `affinity_norm` (z-scored pKd within protocol/antigen) — required because the
  rows pool different assays/scales.
- **Reproduce:** `python scripts/run_regression.py --unified curated_data/unified_affinity_v2.csv --model all --antigen --group-col antigen`

## Artifacts (S3, `s3://scrippsresearch-mshafiei-hackathon/`)
- Embeddings: `embeddings/{igbert,esm2_650M,balm_paired}/unified_raw.npy` (+ antigen, labels)
- Raw results: `results/regression_grouped.json`, `results/regression_random.json`
