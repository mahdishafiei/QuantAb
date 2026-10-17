# Data

Datasets are **not committed to git** (CSVs/PDBs/XLSX are too large). They live in S3 and
the code reads them from a local path. This file documents what we have and how to get it.

## Source of truth — S3 (team bucket in the hackathon AWS account, region `us-west-2`)

```
s3://<your-bucket>/
├── curated_data/
│   ├── unified_affinity_v3.csv  ← USE THIS: 114,300 rows, built from the affinity corpus, normalized
│   ├── unified_affinity_v2.csv  prior version (41k rows, DMS + ANDD build)
│   ├── unified_affinity.csv     older version (no size columns)
│   ├── filtered_data.csv        DMS binding data (Phillips/Dailey/MAGMA-seq)
│   └── ANDD_dataset.xlsx        Terrence's unified Ab/Nb–antigen affinity collection
├── affinity_corpus/             diverse Ab–Ag affinity corpus + identity clustering (2026-06-01)
│   ├── affinity_corpus.parquet           117,486 rows (censored included), full schema
│   ├── affinity_corpus_filtered.parquet  114,652 rows (non-censored only)
│   └── cluster_assignments.parquet       per-row Ab/Ag/joint identity clusters + weights
├── DMS_Data/                    raw deep-mutational-scanning sources
└── embeddings/                  cached model embeddings (one subdir per model)
```

> **v3 (2026-06-01)** is the blessed dataset: built from the affinity corpus by
> `scripts/build_corpus_dataset.py` — measured methods only (predicted KDs dropped), sane KD,
> de-duplicated, and `affinity_norm` = pKd z-scored within each **(method, antigen)** group.
> 114,300 rows · 357 antigens · 0 missing · all paired. Adds the VEGF/AlphaSeq campaign (42k) on
> top of the flu DMS. Antibodies fit the 512 context; embed antigens with `--max-length 1024`
> (only 19 antigens exceed it).

Read access is granted to everyone in the hackathon account, so teammates pull with their
own SSO profile — no shared credentials.

```bash
aws s3 cp s3://<your-bucket>/curated_data/unified_affinity_v3.csv \
  curated_data/ --profile <profile> --region us-west-2
```

## The unified dataset (`unified_affinity_v3.csv`) — what to train on

Built by `scripts/build_corpus_dataset.py` from `affinity_corpus_filtered.parquet`. **114,300
rows**, **357 antigens**, 0 missing, all paired. Each row has an antibody (heavy + light), an
antigen sequence, and a measured affinity — everything needed to inference the three models and
run the affinity regression.

| Column | Meaning |
|---|---|
| `heavy`, `light` | antibody VH / VL |
| `target` | antigen protein sequence |
| `affinity` | **pKd = −log10(KD in M)** (higher = stronger binding) |
| `affinity_norm` | pKd **z-scored within `norm_group`** — the regression target |
| `source` / `dataset` | provenance (phillips_2021, engelhart_2022, …) |
| `antigen`, `gross_antigen`, `method` | antigen names + assay |
| `norm_group` | normalization unit = `method | antigen` |
| `kd_nm`, `kd_sd_nm`, `n_sources_confirmed` | raw KD, uncertainty, cross-source agreement |
| `heavy_len` / `light_len` / `target_len` / `target_truncated` | sequence sizes (truncation flag) |

### Why `affinity_norm`?
Rows pool different assays (yeast display, AlphaSeq/phage, SPR, ELISA) with different scales and
calibration, so raw pKd is **not comparable across them**. `affinity_norm` z-scores within each
`(method, antigen)` group so a single regression sees a consistent target. Use `affinity_norm` as
the label; `affinity` (raw pKd) is kept for reference. (Composition is Flu+VEGF-dominated and ~250
antigens are singletons → their `affinity_norm` is 0.)

### What was dropped / excluded
- **Predicted KDs** (`method=Predict`) and physically-absurd values; kept only measured methods + sane KD (1 mM–0.1 pM).
- Not usable elsewhere: PLAbDab (no labels), SARS2 escape (escape ≠ KD), IEDB epitopes (categorical),
  AB-Bind (ΔΔG), SKEMPI (no sequences), SAbDab-affinity file (Y all 0).

## Rebuild it
```bash
python scripts/build_corpus_dataset.py \
  --corpus affinity_corpus/affinity_corpus_filtered.parquet \
  --out    curated_data/unified_affinity_v3.csv
```
(Prior DMS+ANDD build — v2 — via `scripts/build_unified_affinity.py`.)

## Programmatic load
```python
from pathlib import Path
from quantab.data import load_unified
df = load_unified(Path("curated_data/unified_affinity_v3.csv"))
```

## The diverse affinity corpus (`affinity_corpus/`) — next-gen training data

Built by the `sourcing_affinities/` pipeline (`01_clean_sources.py` → `02_aggregate.py` →
`03_cluster.py`), a from-scratch re-curation that supersedes the DMS-only `unified_affinity`
set with a more diverse, multi-source corpus. **It is the basis for `unified_affinity_v3.csv`
(see above) — embeddings extracted and benchmarked; results in [`results-classic.md`](../results-classic.md).**

| File | Rows | Notes |
|---|---|---|
| `affinity_corpus.parquet` | 117,486 | full schema; includes censored rows (`is_censored=True` for KD > 1 µM) |
| `affinity_corpus_filtered.parquet` | 114,652 | non-censored only |
| `cluster_assignments.parquet` | 117,486 | identity-cluster labels + weights, joinable on `hc_seq`/`lc_seq`/`target_seq` |

**Corpus schema** (`affinity_corpus.parquet`): `hc_seq`, `lc_seq`, `target_seq`, `kd_nm`,
`kd_sd_nm`, `source_key`, `source_citation`, `method`, `gross_antigen`, `fine_antigen`,
`is_censored`, `censor_type`, `notes`, plus `affinity = -log10(kd_nm * 1e-9)`.

**Sources:** phillips_2021 (67,033, Flu), engelhart_2022 (42,382, VEGF-A), dailey_2021 (4,974,
Flu), andd_2026 (4,302, diverse), shanehsazzadeh_2023 (422, HER2), magma_seq_2022 (153, SARS-CoV-2 RBD).

**Clustering** (`cluster_assignments.parquet`, 38 cols) — for leakage-free held-out splits:
- Antibody (heavy chain): `hc_cluster_{40,70,90,95,97}` + matching `hc_weight_*`
- Antigen: `ag_cluster_{80,95}` + `ag_weight_*`
- Joint Ab×Ag: `joint_ab{40,70,90,95,97}_ag{80,95}` + `joint_weight_*` (inverse-cluster-size weights)

All threshold columns are retained — the primary split for GroupKFold/embedding is **TBD** (decide
once embeddings exist).

```bash
aws s3 cp s3://<your-bucket>/affinity_corpus/affinity_corpus_filtered.parquet \
  sourcing_affinities/outputs/ --profile <profile> --region us-west-2
```

## Source provenance
| Dataset | Origin |
|---|---|
| Phillips 2021 | CR9114/CR6261 anti-influenza DMS (Tite-seq), *eLife* |
| Dailey | CR9114 single-mutant KD tables |
| MAGMA-seq | 4A8 / CC12.1 vs SARS-CoV-2 RBD (yeast display) |
| ANDD | Terrence's merge of SAbDab, sdAb-DB, abYbank, PDB, PLAbDab-nano, INDI, … (literature/SPR Kd). Not public: its rows are excluded from the committed `sourcing_affinities/` parquets. |

All datasets derive from published studies and retain their original licenses. The S3 bucket
is a convenience mirror for the hackathon, not a re-publication.
