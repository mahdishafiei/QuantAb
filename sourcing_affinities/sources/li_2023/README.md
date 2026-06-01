# Li et al. 2023 — ML-generated scFv variants (aayl49_ML)

## Citation
Li et al. (2023) Machine learning optimization of candidate antibody yields
high-affinity anti-VEGF antibodies. *Cell Reports Methods* [check full citation].

## Description
ML-generated scFv variants based on the aayl49 scaffold (anti-VEGF-A, same
template as Engelhart 2022). Variants tested by phage display enrichment.
AbBiBench ID: aayl49_ML. PDB: none (same antigen as Engelhart aayl49).

## Download Instructions

1. Check the paper supplement for the ML variant dataset.
2. May be available on the Engelhart Lab GitHub or paper Zenodo.
3. Search: "aayl49_ML_benchmarking_data.csv" — this file may already be present
   in DMS_Data/05_engelhart_aayl/ (check before downloading separately).

```bash
ls ../../../../DMS_Data/05_engelhart_aayl/aayl49_ML_benchmarking_data.csv
```

If present, this source can share the `source_engelhart.py` loader (it already
processes aayl49_ML files if they follow the same schema). In that case, this
stub is not needed as a separate source.

## Status
⚠️  Check if already processed by engelhart_2022 source script. If not,
place raw data in `raw/` and implement `source_li.py`.
