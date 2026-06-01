# Warszawski et al. 2020 — D44.1 anti-lysozyme yeast display DMS (PDB 1MLC)

## Status: ✅ INGESTED via AbBiBench (`sources/abbibench/`)

Ingested from the AbBiBench HuggingFace release (`1mlc_benchmarking_data.csv`),
not a standalone download. The `source_warszawski.py` stub is retained for
reference.

## Citation
Warszawski S et al. (2020) Optimizing antibody affinity and stability by the
automated design of the variable light-heavy chain interfaces.
*PLOS Comput Biol* 16(4):e1007828. doi:10.1371/journal.pcbi.1007828

## Correction
Antibody is **D44.1**; antigen is **hen egg-white lysozyme** (PDB 1MLC chain E,
129 aa) — *not* anti-VEGF as an earlier draft of this stub stated.

## Measurement
`binding_score` = log enrichment from yeast display — **not** calibrated KD.
Ingested as `measure_type="log_enrichment"` with `kd_nm`/`affinity` left null.
