# Koenig et al. 2017 — G6.31 anti-VEGF Fab DMS (PDB 2FJG)

## Status: ✅ INGESTED via AbBiBench (`sources/abbibench/`)

Ingested from the AbBiBench HuggingFace release (`2fjg_benchmarking_data.csv` and
`g6_LC_benchmarking_data.csv`), not a standalone download. The `source_koenig.py`
stub is retained for reference.

## Citation
Koenig P et al. (2017) Mutational landscape of antibody variable domains reveals
a switch between stability and binding optimization at cold temperatures.
*PNAS* 114(3):497-502. doi:10.1073/pnas.1613231114

## Correction
Antibody is **G6.31** (anti-VEGF-A Fab; ranibizumab is a related lineage).
Antigen VEGF-A is PDB 2FJG chain V (~95 aa).

## Measurement
`binding_score` = log enrichment from phage display — **not** calibrated KD.
Ingested as `measure_type="log_enrichment"` with `kd_nm`/`affinity` left null;
use `binding_score` for within-dataset rank/Spearman only.
