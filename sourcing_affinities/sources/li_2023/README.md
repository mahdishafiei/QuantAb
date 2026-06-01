# Li et al. 2023 — ML-generated scFv variants (aayl49_ML)

## Status: ✅ ALREADY IN CORPUS via engelhart_2022 — do not ingest separately

Verified 2026-06-01: the AbBiBench `aayl49_ML_benchmarking_data.csv` (8,953 rows)
is **100% sequence-overlapping** with our existing `engelhart_2022` source. The
same holds for `aayl50`/`aayl52`. No separate ingestion is needed; doing so would
duplicate rows. The `source_li.py` stub is retained for reference only.

## Citation
Li et al. (2023) Machine learning optimization of candidate antibody yields
high-affinity anti-SARS-CoV-2 antibodies. *Cell Reports Methods*.

## Note on antigen
The AAYL49/50/51/52 lineages (Engelhart 2022 AlphaSeq) bind **SARS-CoV-2 Spike**,
not VEGF as an earlier draft stated. (Our `engelhart_2022` rows are currently
mislabeled `gross_antigen=VEGF-A` — flagged for a separate QC pass.)
