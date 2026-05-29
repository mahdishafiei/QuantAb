# Data

The DMS datasets are **not committed to git** (~162 MB of CSVs, PDBs, and raw
flow-cytometry files). They live in S3 and on the server, and the code reads
them from a local path. This file documents where to get them.

## Source of truth — S3 (hackathon account)

```
s3://scrippsresearch-mshafiei-hackathon/DMS_Data/
s3://scrippsresearch-mshafiei-hackathon/AB-Bind-Database/
```

Region: `us-west-2`. Read access (`s3:GetObject`, `s3:ListBucket`) is granted
to every member of the Scripps Hackathon AWS account (`127696279288`), so any
teammate can pull it with their own SSO profile — no shared credentials.

### Download to the default local path

```bash
# From a laptop / dev box (replace <profile> with your SSO profile name)
aws s3 sync s3://scrippsresearch-mshafiei-hackathon/DMS_Data ./DMS_Data \
  --profile <profile> --region us-west-2

# From an EC2 instance with the hackathon-ec2-profile (no --profile needed)
aws s3 sync s3://scrippsresearch-mshafiei-hackathon/DMS_Data /data/DMS_Data \
  --region us-west-2
```

Point the pipeline at wherever you put it:

```python
from pathlib import Path
from quantab.data import load_all, summarize

DATA_DIR = Path("/home/jovyan/work/Hackaton/DMS_Data")  # or /data/DMS_Data on EC2
df = load_all(DATA_DIR)
summarize(df)
```

## Datasets

| Directory | Dataset | Measurement | N variants | Origin |
|---|---|---|---|---|
| `02_phillips_cr9114_cr6261/` | Phillips et al. 2021 CR9114 anti-influenza DMS | `*_neg_log_kd` | ~67,000 | Phillips et al., *eLife* 2021 |
| `05_engelhart_aayl/` | Engelhart et al. AAYL variants | `binding_score` | ~varies | Engelhart et al. |
| `06_shanehsazzadeh_trastuzumab/` | Shanehsazzadeh trastuzumab zero-shot binders | `-log(KD (M))` | 422 | Shanehsazzadeh et al. |
| `01_dailey_cr9114/` | Dailey CR9114 single-point mutants | `minus_log_Kd` | ~varies | Dailey et al. |
| `03_magma_seq/` | MAGMA-seq data | mixed | — | — |
| `04_adams_4420/` | Adams et al. titeseq (raw FCS + replicates) | titeseq | — | Adams et al. |
| `pdbs/` | Reference antibody-antigen structures | — | — | RCSB PDB |

`AB-Bind-Database/` (ΔΔG mutation data + homology-model PDBs) is an additional
antibody-binding resource, mirrored from
[sarahsirin/AB-Bind-Database](https://github.com/sarahsirin/AB-Bind-Database).

> All datasets are derived from published studies and retain their original
> licenses. This bucket is a convenience mirror for the hackathon, not a
> re-publication.
