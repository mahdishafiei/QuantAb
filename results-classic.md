# results-classic — Classical affinity regression across antibody language models

**One line:** we predict antibody–antigen binding affinity from three protein/antibody language
models' embeddings and compare them; on the honest evaluation (unseen antigens) the
antibody-specific **IgBERT wins on every metric**.

---

## 1. Task
Given an antibody (heavy + light chains) and an antigen sequence, predict the **binding affinity**
(KD). This is a **regression** — the model outputs a continuous affinity; a "does it bind" answer
is just a threshold on that. We use it to **benchmark the three language models** as feature
extractors.

## 2. Dataset — `unified_affinity_v3.csv`
`s3://<your-bucket>/curated_data/unified_affinity_v3.csv` — **114,300 rows,
0 missing, all paired**. Each row = one antibody–antigen pair with a measured affinity. Built from
the team's affinity-sourcing corpus, cleaned + normalized (`scripts/build_corpus_dataset.py`).

| Source | System | Rows | Antigen |
|---|---|---|---|
| Phillips 2021 | CR9114 anti-influenza DMS | 67,033 | flu HA |
| Engelhart 2022 (AlphaSeq) | anti-VEGF campaign | 42,381 | VEGF |
| Dailey 2021 | CR9114 single-mutants | 3,699 | flu HA |
| ANDD 2026 | diverse SAbDab / PDB / sdAb | 702 | many |
| Shanehsazzadeh 2023 | trastuzumab | 421 | HER2 |
| MAGMA-seq 2022 | 4A8 / CC12.1 | 64 | SARS-CoV-2 |

**357 unique antigens · 54,434 unique heavy chains · pKd 6.0–12.4.** Measured methods only
(yeast display 70.8k, AlphaSeq/phage 42.4k, SPR 440, ELISA 50; predicted KDs dropped).
**Coverage is uneven and 2-antigen-dominated** (Flu + VEGF ≈ 99% of rows; 250 antigens are
singletons).

**Label.** `affinity` = pKd (−log10 KD). **`affinity_norm`** = pKd **z-scored within each
(method, antigen) group** — the regression target. Pooling assays with different scales/calibration
(yeast display vs AlphaSeq vs SPR) makes raw KD non-comparable; per-group normalization removes the
assay/antigen offset so the target is comparable across the corpus.

## 3. Models and exactly how sequences are fed
Frozen feature extractors; embeddings **mean-pooled over residue positions** (special tokens
excluded — CLS not used).

| Model | Type | Dim | How VH + VL are input (decoded from the tokenizer) |
|---|---|---|---|
| **IgBERT** (`Exscientia/IgBert`) | antibody-specific BERT | 1024 | `[CLS] <VH><VL> [SEP]` — concatenated, no separator |
| **ESM-2 650M** (`facebook/esm2_t33_650M_UR50D`) | general protein | 1280 | `<cls> <VH><VL> <eos>` — concatenated, no separator |
| **BALM-paired** (`brineylab/BALM-paired`) | natively-paired antibody | 1024 | `<s> VH </s></s> VL </s>` — native pair, `</s></s>` = chain boundary |

The **antigen** is embedded separately with ESM-2 650M (mean-pooled, 1024-token context,
de-duplicated over unique antigens).

## 4. Pipeline
```
sequences → language model → mean-pool → (raw embedding) → StandardScaler → MLP → predicted affinity
```
No PCA on the classical side (full embedding used; PCA only exists for the quantum qubit budget).
Scaling fit inside each CV fold (no leakage).

## 5. Evaluation
**5-fold cross-validation holding out whole antigens** (`GroupKFold` by `antigen`) — each model is
scored on antigens it never saw in training. Target = `affinity_norm`.

## 6. Results (held-out antigens, antibody features)

| Model | Spearman ρ ↑ | Pearson r ↑ | R² ↑ | RMSE ↓ | Recall@10% ↑ |
|---|---|---|---|---|---|
| **IgBERT** | **0.31** | **0.26** | **−0.39** | **1.12** | **0.14** |
| BALM-paired | 0.23 | 0.15 | −0.28 | 1.08 | 0.12 |
| ESM-2 650M | 0.16 | 0.12 | −0.35 | 1.12 | 0.12 |

**Metrics:** Spearman/Pearson = rank/linear correlation; R² = variance explained (negative = worse
than predicting the mean); RMSE = error (normalized units, lower better); Recall@10% = share of the
truly strongest binders found in the model's top 10% (random ≈ 0.10).

## 7. Takeaways
- **IgBERT (antibody-specific) is best on every correlation/retrieval metric.** Antibody-specific
  pretraining beats the general protein model (ESM-2) and the paired model (BALM) here.
- **Generalizing to unseen antigens is hard.** All R² are negative (no model beats predicting the
  mean in absolute terms) and recall@10% is ~0.12–0.14 — modest, reflecting the difficulty and the
  thin per-antigen coverage, not a bug. Ranking signal (Spearman) is the usable part.
- **The antigen embedding doesn't help** (Spearman drops for IgBERT/BALM when added) — the data is
  Flu+VEGF-dominated, so the antigen is near-constant and adds noise.
- Scores are lower than the earlier (smaller) v2 dataset because v3 is larger, more diverse, and the
  held-out-antigen test over 357 groups is genuinely harder.

## 8. Limitations
- IgBERT and ESM-2 get VH+VL with no chain-boundary token (concatenated); BALM gets a native pair.
- Two antigen families (Flu, VEGF) dominate; 250 antigens are singletons (→ `affinity_norm` 0, no
  within-group signal) — caps how well cross-antigen generalization can be measured.
- `affinity_norm` is comparable within a (method, antigen) group, not as an absolute cross-assay KD
  (a deliberate choice — see the normalization discussion).

## 9. Reproduce / artifacts
```bash
python scripts/build_corpus_dataset.py    # corpus -> unified_affinity_v3.csv (clean + normalize)
python scripts/extract_embeddings.py --unified curated_data/unified_affinity_v3.csv --model all --embed antibody
python scripts/run_regression.py    --unified curated_data/unified_affinity_v3.csv --model all --group-col antigen
```
- Dataset: `s3://<your-bucket>/curated_data/unified_affinity_v3.csv`
- Embeddings: `s3://<your-bucket>/embeddings/{igbert,esm2_650M,balm_paired}/`
- Scores: `s3://<your-bucket>/results/regression_grouped.json`
