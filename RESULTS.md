# Results — affinity regression

Three antibody language models, mean-pooled embeddings → MLP regression on
`unified_affinity_v2.csv` (40,937 rows, 1,041 antigens). Target: binding affinity
(−log10 KD, normalized per assay/antigen). ρ = Spearman correlation between predicted and
measured affinity.

| Model | ρ (antibody) | ρ (antibody + antigen) | ρ (random split, ref.) |
|---|---|---|---|
| **IgBERT** | **0.48 ± 0.13** | 0.44 ± 0.14 | 0.90 ± 0.00 |
| ESM-2 650M | 0.45 ± 0.18 | 0.45 ± 0.19 | 0.91 ± 0.00 |
| BALM-paired | 0.42 ± 0.11 | 0.38 ± 0.23 | 0.89 ± 0.01 |

## What the columns mean
- **ρ (antibody)** — the headline result. Features = the model's antibody embedding only.
  5-fold cross-validation **holding out whole antigens**, so the model is tested on antigens it
  never saw during training. IgBERT is best; the high ±std reflects how different held-out
  antigens are.
- **ρ (antibody + antigen)** — same setup, but the ESM-2 antigen embedding is concatenated to the
  antibody features. It does **not** help (and sometimes hurts), because the dataset is dominated
  by one antigen family (influenza), so the antigen signal is near-constant.
- **ρ (random split, ref.)** — rows shuffled into folds instead of grouped by antigen. Much higher
  (~0.9) because ~80% of rows are single-mutant CR9114 variants, so test sequences closely
  resemble training ones. Shown only for reference — it overstates true generalization.

## Method
- Mean pooling over residue positions (excludes special tokens); CLS rejected.
- Feature scaling fit **inside each fold** (no leakage); classical regressor (MLP head), no PCA.
- Reproduce: `python scripts/run_regression.py --unified curated_data/unified_affinity_v2.csv --model all --antigen --group-col antigen`

Raw scores: `s3://scrippsresearch-mshafiei-hackathon/results/regression_{grouped,random}.json`
