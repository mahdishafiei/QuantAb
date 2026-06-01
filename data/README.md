# Data

Datasets are **not committed to git** (CSVs/PDBs/XLSX are too large). They live in S3 and
the code reads them from a local path. This file documents what we have and how to get it.

## Source of truth — S3 (hackathon account `127696279288`, region `us-west-2`)

```
s3://scrippsresearch-mshafiei-hackathon/
├── curated_data/
│   ├── unified_affinity_v2.csv  ← USE THIS: ready-to-go, normalized, with size metadata
│   ├── unified_affinity.csv     prior version (kept; no size columns)
│   ├── filtered_data.csv        DMS binding data (Phillips/Dailey/MAGMA-seq)
│   └── ANDD_dataset.xlsx        Terrence's unified Ab/Nb–antigen affinity collection
├── affinity_corpus/             diverse Ab–Ag affinity corpus + identity clustering (2026-06-01)
│   ├── affinity_corpus.parquet           117,486 rows (censored included), full schema
│   ├── affinity_corpus_filtered.parquet  114,652 rows (non-censored only)
│   └── cluster_assignments.parquet       per-row Ab/Ag/joint identity clusters + weights
├── DMS_Data/                    raw deep-mutational-scanning sources
└── embeddings/                  cached model embeddings (one subdir per model)
```

> **v2 (2026-05-31)** is the blessed ready-to-go version: identical rows to v1, audited
> (0 missing, 0 dup sequence-triples, `affinity_norm` finite), plus `heavy_len / light_len /
> target_len / target_truncated` size columns. Antibodies fully fit the 512-token context
> (max combined 471); embed antigens with `--max-length 1024` — only **146 / 40,937** antigens
> still exceed that.

Read access is granted to everyone in the hackathon account, so teammates pull with their
own SSO profile — no shared credentials.

```bash
aws s3 cp s3://scrippsresearch-mshafiei-hackathon/curated_data/unified_affinity.csv \
  curated_data/ --profile <profile> --region us-west-2
```

## The unified dataset (`unified_affinity.csv`) — what to train on

Built by `scripts/build_unified_affinity.py` from `filtered_data.csv` + the usable subset of
ANDD. **40,937 rows**, **1,041 antigens**. Each row has an antibody (heavy + light), an
antigen sequence, and an affinity label — everything needed to inference the three models and
run an affinity regression.

| Column | Meaning |
|---|---|
| `heavy`, `light` | antibody VH / VL (light empty for nanobodies / heavy-only records) |
| `target` | antigen protein sequence |
| `affinity` | **pKd = −log10(KD in M)** (higher = stronger binding) |
| `affinity_norm` | pKd **z-scored within `norm_group`** — the regression target |
| `source`, `antigen` | provenance + antigen name |
| `norm_group` | normalization unit (per source×antigen for DMS; one block for ANDD) |
| `dataset` | `phillips2021` / `dailey` / `magma_seq` / `ANDD` |
| `heavy_len` / `light_len` / `target_len` | sequence lengths (residues) — v2 |
| `target_truncated` | True if antigen > 1024 tokens (truncated when embedded) — v2 |

### Why `affinity_norm`?
The rows come from different assays/protocols (Tite-seq, KD tables, MAGMA-seq yeast display,
and ANDD's literature/SPR Kd) with different scales and dynamic-range ceilings, so raw pKd is
**not comparable across rows**. `affinity_norm` z-scores within each protocol/antigen group so
a single regression sees a consistent target. Use `affinity_norm` as the label; `affinity`
(raw pKd) is kept for reference.

### What was dropped (not usable for affinity regression)
- Phillips CR9114 **H3-HA** — ~88% of values pinned at the assay ceiling (degenerate).
- ANDD rows without a heavy chain, antigen sequence, or a sane measured KD (1 pM–1 mM); predicted KDs excluded.
- From Terrence's bucket: PLAbDab (sequences only, no labels), SARS2 escape (escape ≠ KD),
  IEDB epitopes (categorical), AB-Bind (ΔΔG, a different quantity).
- **New affinity files (assessed 2026-05-31, not folded in):** `affinity/antibody_affinity_sabdab.csv`
  has its `Y` label all 0.0 (no usable values); `affinity/skempi_v2.csv` is PDB+mutation only
  (no sequences without resolving structures); `affinity/alphaseq/...` has antibody seqs +
  affinity but **no antigen sequence** (target is a name) — usable for antibody-only later, not
  for the two-modality setup.

## Rebuild it
```bash
python scripts/build_unified_affinity.py \
  --curated curated_data/filtered_data.csv \
  --andd    curated_data/ANDD_dataset.xlsx \
  --out     curated_data/unified_affinity_v2.csv
```

## Programmatic load
```python
from pathlib import Path
from quantab.data import load_unified
df = load_unified(Path("curated_data/unified_affinity_v2.csv"))
```

## The diverse affinity corpus (`affinity_corpus/`) — next-gen training data

Built by the `sourcing_affinities/` pipeline (`01_clean_sources.py` → `02_aggregate.py` →
`03_cluster.py`), a from-scratch re-curation that supersedes the DMS-only `unified_affinity`
set with a more diverse, multi-source corpus. **Pushed to S3 2026-06-01; pending embedding
extraction by Mahdi.**

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
aws s3 cp s3://scrippsresearch-mshafiei-hackathon/affinity_corpus/affinity_corpus_filtered.parquet \
  sourcing_affinities/outputs/ --profile <profile> --region us-west-2
```

## Source provenance
| Dataset | Origin |
|---|---|
| Phillips 2021 | CR9114/CR6261 anti-influenza DMS (Tite-seq), *eLife* |
| Dailey | CR9114 single-mutant KD tables |
| MAGMA-seq | 4A8 / CC12.1 vs SARS-CoV-2 RBD (yeast display) |
| ANDD | Terrence's merge of SAbDab, sdAb-DB, abYbank, PDB, PLAbDab-nano, INDI, … (literature/SPR Kd) |

All datasets derive from published studies and retain their original licenses. The S3 bucket
is a convenience mirror for the hackathon, not a re-publication.
