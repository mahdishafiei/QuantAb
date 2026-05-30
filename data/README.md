# Data

Datasets are **not committed to git** (CSVs/PDBs/XLSX are too large). They live in S3 and
the code reads them from a local path. This file documents what we have and how to get it.

## Source of truth — S3 (hackathon account `127696279288`, region `us-west-2`)

```
s3://scrippsresearch-mshafiei-hackathon/
├── curated_data/
│   ├── unified_affinity.csv     ← USE THIS: merged, normalized, regression-ready
│   ├── filtered_data.csv        DMS binding data (Phillips/Dailey/MAGMA-seq)
│   └── ANDD_dataset.xlsx        Terrence's unified Ab/Nb–antigen affinity collection
├── DMS_Data/                    raw deep-mutational-scanning sources
└── embeddings/                  cached model embeddings (one subdir per model)
```

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

## Rebuild it
```bash
python scripts/build_unified_affinity.py \
  --curated curated_data/filtered_data.csv \
  --andd    curated_data/ANDD_dataset.xlsx \
  --out     curated_data/unified_affinity.csv
```

## Programmatic load
```python
from pathlib import Path
from quantab.data import load_unified
df = load_unified(Path("curated_data/unified_affinity.csv"))
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
