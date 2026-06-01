# Clark et al. 2006 — AQC2 anti-integrin-α1 affinity maturation (PDB 1MHP)

## Status: ✅ INGESTED via AbBiBench (`sources/abbibench/`)

Ingested from the AbBiBench HuggingFace release (`1mhp_LC_benchmarking_data.csv`),
not a standalone download. The `source_clark.py` stub is retained for reference.

## Citation
Clark LA et al. (2006) Affinity enhancement of an in vivo matured therapeutic
antibody using structure-based computational design. *Protein Sci* 15(5):949-960.
doi:10.1110/ps.052030506

## Correction
Antibody is **AQC2**; antigen is the **integrin α1 I-domain** (PDB 1MHP chain A,
~184 aa) — *not* anti-VEGF as an earlier draft of this stub stated.

## Measurement
`binding_score` = −log10(KD). Ingested as `measure_type="kd"`,
`kd_nm = 10^(9 − binding_score)`. ~69 variants (LC scan).
