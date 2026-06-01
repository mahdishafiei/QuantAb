# Results — affinity regression

Three antibody language models, mean-pooled embeddings → MLP regression on
`unified_affinity_v2.csv` (40,937 rows, 1,041 antigens). Target: binding affinity
(−log10 KD, normalized per assay/antigen).

| Model | Spearman ρ |
|---|---|
| IgBERT | **0.48 ± 0.13** |
| ESM-2 650M | 0.45 ± 0.18 |
| BALM-paired | 0.42 ± 0.11 |

Spearman correlation between predicted and measured affinity, 5-fold cross-validation
**holding out whole antigens** (the model is tested on antigens it never saw in training).

**Method:** mean pooling over residues; feature scaling fit inside each fold (no leakage).
Adding the antigen embedding did not improve results on this (flu-dominated) dataset.

**Reproduce:** `python scripts/run_regression.py --unified curated_data/unified_affinity_v2.csv --model all --group-col antigen`

Raw scores: `s3://scrippsresearch-mshafiei-hackathon/results/regression_grouped.json`
