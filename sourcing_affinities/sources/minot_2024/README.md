# Minot & Reddy 2024 — 5A12 anti-Ang2 / anti-VEGF yeast display DMS

## Status: ✅ INGESTED via AbBiBench (`sources/abbibench/`)

Ingested from the AbBiBench HuggingFace release (`5a12_ang2_benchmarking_data.csv`
and `5a12_vegf_benchmarking_data.csv`), not a standalone download. The
`source_minot.py` stub is retained for reference.

## Citation
Minot M & Reddy ST (2024) Deep mutational scanning of antibody 5A12 against
Angiopoietin-2 / VEGF.

## Systems
- **5A12 / Angiopoietin-2** — antigen PDB 4ZFG chain A (~220 aa, FLD).
- **5A12 / VEGF-A** — antigen PDB 4ZFF chain D (~99 aa). 5A12 is cross-reactive.

## Measurement
`binding_score` = log enrichment from yeast display — **not** calibrated KD.
Ingested as `measure_type="log_enrichment"` with `kd_nm`/`affinity` left null.
The VEGF set is large (~30k variants).
