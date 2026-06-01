# Results — affinity regression

Three antibody language models, mean-pooled embeddings → MLP regression on
`unified_affinity_v2.csv` (40,937 rows, 1,041 antigens). Target: binding affinity
(−log10 KD, normalized per assay/antigen). Evaluation: 5-fold cross-validation **holding out
whole antigens** — each model is scored on antigens it never saw during training.

| Model | Spearman ρ ↑ | Pearson r ↑ | R² ↑ | RMSE ↓ | Recall@10% ↑ |
|---|---|---|---|---|---|
| **IgBERT** | **0.51** | **0.51** | **0.10** | **0.93** | **0.24** |
| ESM-2 650M | 0.38 | 0.38 | −0.54 | 1.11 | 0.23 |
| BALM-paired | 0.38 | 0.37 | −0.10 | 1.02 | 0.18 |

## Metrics (what they measure)
- **Spearman ρ** — rank correlation; do higher-predicted antibodies actually bind tighter. Primary metric for affinity/fitness tasks.
- **Pearson r** — linear correlation of predicted vs measured affinity.
- **R²** — fraction of variance explained; **negative means worse than predicting the dataset mean** (i.e. no useful signal on held-out antigens).
- **RMSE** — root-mean-square error in normalized affinity (std) units; lower is better.
- **Recall@10%** — of the truly strongest binders (top 10% by measured affinity), the fraction the model ranks in its own top 10%. Random ≈ 0.10. This is the practically useful number for screening / antibody engineering.

## Takeaways
- **IgBERT (antibody-specific) wins on every metric** and is the only model with positive R² —
  on unseen antigens, ESM-2 and BALM do no better than predicting the mean.
- **Generalizing to new antigens is hard.** Even the best model recovers only ~24% of the top
  binders; the modest scores reflect the difficulty, not a bug.
- **The antigen embedding doesn't help** (sometimes hurts) — the data is dominated by one antigen
  family (influenza), so the antigen signal is near-constant.
- **Reference:** on a plain random split (rows shuffled, not grouped by antigen) all models reach
  ρ ≈ 0.9 — but ~80% of rows are single-mutant CR9114 variants, so test sequences closely resemble
  training. That number overstates real generalization and is not the one to report.

## Method
- Mean pooling over residue positions (excludes special tokens); CLS rejected.
- Feature scaling fit **inside each fold** (no leakage); MLP head; no PCA.
- Reproduce: `python scripts/run_regression.py --unified curated_data/unified_affinity_v2.csv --model all --group-col antigen`

Raw scores: `s3://scrippsresearch-mshafiei-hackathon/results/regression_grouped.json`
