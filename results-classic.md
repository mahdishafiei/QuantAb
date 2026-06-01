# results-classic — Classical affinity regression across antibody language models

**One line:** we predict antibody–antigen binding affinity from three protein/antibody language
models' embeddings and compare them; on the honest evaluation (unseen antigens) the
antibody-specific **IgBERT wins on every metric**.

---

## 1. Task
Given an antibody (heavy + light chains) and an antigen sequence, predict the **binding affinity**
(KD). This is a **regression** — the model outputs a continuous affinity; a "does it bind"
answer is just a threshold on that prediction. We use it to **benchmark the three language
models** as feature extractors.

## 2. Dataset — `unified_affinity_v2.csv`
`s3://scrippsresearch-mshafiei-hackathon/curated_data/unified_affinity_v2.csv` — **40,937 rows,
0 missing**. Each row = one antibody–antigen pair with a measured affinity.

| Column | Meaning |
|---|---|
| `heavy`, `light` | antibody VH / VL (light empty for nanobodies) |
| `target` | antigen protein sequence |
| `affinity` | **pKd = −log10(KD in M)** (higher = binds tighter) |
| `affinity_norm` | pKd **z-scored within each (protocol, antigen) group** → the regression target |
| `source`, `antigen`, `dataset` | provenance |
| `heavy_len / light_len / target_len / target_truncated` | sequence sizes (truncation flag) |

**Composition:** Phillips CR9114 vs flu (33,345) · Dailey CR9114 (4,196) · ANDD diverse SAbDab/PDB/sdAb (3,310) · MAGMA-seq SARS-CoV-2 (86). **1,041 unique antigens**, **38,066 unique heavy chains**, pKd range 3.0–12.0.

**Every row has a KD** (it was an entry filter). But coverage per antigen is very uneven:
**687/1,041 antigens have only one measurement**; the top 3 antigens (flu H1/H3, HER2) are **93%**
of all rows. So it's deep on flu/CR9114 and a thin, mostly-singleton tail elsewhere.

**Why normalize?** Rows pool different assays (Tite-seq, KD tables, yeast display, literature SPR)
with different scales/ceilings, so raw pKd isn't comparable across them. `affinity_norm`
(z-scored per protocol/antigen) puts everything on one comparable target.

## 3. Models and exactly how sequences are fed
All three are run as frozen feature extractors; embeddings are **mean-pooled over residue
positions** (special tokens excluded — the CLS token is *not* used; see Methods).

| Model | Type | Dim | How VH + VL are input (decoded from the tokenizer) |
|---|---|---|---|
| **IgBERT** (`Exscientia/IgBert`) | antibody-specific BERT | 1024 | `[CLS] <VH><VL> [SEP]` — chains **concatenated, no separator** |
| **ESM-2 650M** (`facebook/esm2_t33_650M_UR50D`) | general protein | 1280 | `<cls> <VH><VL> <eos>` — concatenated, no separator |
| **BALM-paired** (`brineylab/BALM-paired`) | natively-paired antibody | 1024 | `<s> VH </s></s> VL </s>` — **native pair**, `</s></s>` marks the chain boundary |

The **antigen** is embedded separately with ESM-2 650M (mean-pooled, 1024-token context,
de-duplicated over the 1,041 unique antigens). Concatenated to the antibody features it forms the
"antibody + antigen" variant.

## 4. Pipeline
```
sequences → language model → mean-pool → (raw embedding) → StandardScaler → MLP → predicted affinity
```
- **No PCA on the classical side** — the regressor uses the full embedding (1024/1280-dim). (PCA
  only exists for the quantum-kernel arm, to fit the qubit budget.)
- **No leakage:** the scaler is fit inside each cross-validation fold (train only).

## 5. Evaluation
- **5-fold cross-validation holding out whole antigens** (`GroupKFold` by `antigen`) — the model is
  always scored on antigens it never saw in training. This is the honest generalization estimate.
- Target = `affinity_norm`.

## 6. Results (held-out antigens)

| Model | Spearman ρ ↑ | Pearson r ↑ | R² ↑ | RMSE ↓ | Recall@10% ↑ |
|---|---|---|---|---|---|
| **IgBERT** | **0.51** | **0.51** | **0.10** | **0.93** | **0.24** |
| ESM-2 650M | 0.38 | 0.38 | −0.54 | 1.11 | 0.23 |
| BALM-paired | 0.38 | 0.37 | −0.10 | 1.02 | 0.18 |

**Metrics:** Spearman/Pearson = rank/linear correlation of predicted vs measured affinity; R² =
variance explained (negative = worse than predicting the mean); RMSE = error (normalized units,
lower better); Recall@10% = share of the truly strongest binders found in the model's top 10%
(random ≈ 0.10).

Reference: on a **random** split (rows shuffled, not grouped by antigen) all models reach ρ ≈ 0.9 —
but ~80% of rows are single-mutant CR9114 variants, so test sequences closely resemble training.
That number overstates real generalization and is **not** the headline.

## 7. Takeaways
- **IgBERT (antibody-specific) is best on every metric**, and the only model with positive R² —
  on unseen antigens, the general protein model (ESM-2) and the paired model (BALM) don't beat
  predicting the mean for absolute affinity, though all three rank somewhat above chance.
- **Generalizing across antigens is hard:** the best model recovers only ~24% of the top binders.
  Modest scores reflect the difficulty (and the thin per-antigen coverage), not a bug.
- **The antigen embedding doesn't help** — the data is flu-dominated and most antigens appear once,
  so there's little cross-antigen signal to exploit.

## 8. Limitations / honest caveats
- **IgBERT and ESM-2 receive VH+VL with no chain-boundary token** (just concatenated). BALM gets a
  proper native pair. Giving IgBERT/ESM-2 an explicit separator could change their numbers — a
  worthwhile follow-up.
- **Antigen coverage is shallow** (median 1 measurement/antigen), and the set is **flu-heavy**, which
  caps how well cross-antigen generalization can be measured.
- **Normalized target:** `affinity_norm` is comparable within a protocol/antigen group, not as an
  absolute cross-assay KD — by design.
- Negative R² for ESM-2/BALM = ranking is okay but absolute predictions are miscalibrated for unseen
  antigens (Spearman/Recall are the more meaningful comparison here).

## 9. Reproduce / artifacts
```bash
# embeddings (GPU): python scripts/extract_embeddings.py --unified curated_data/unified_affinity_v2.csv --model all --embed antibody
# regression:       python scripts/run_regression.py    --unified curated_data/unified_affinity_v2.csv --model all --group-col antigen
```
- Dataset: `s3://scrippsresearch-mshafiei-hackathon/curated_data/unified_affinity_v2.csv`
- Embeddings: `s3://scrippsresearch-mshafiei-hackathon/embeddings/{igbert,esm2_650M,balm_paired}/`
- Scores: `s3://scrippsresearch-mshafiei-hackathon/results/regression_grouped.json`
- This file: `s3://scrippsresearch-mshafiei-hackathon/results/results-classic.md`
