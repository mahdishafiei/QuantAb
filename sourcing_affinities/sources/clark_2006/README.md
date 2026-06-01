# Clark et al. 2006 — Computationally-directed anti-VEGF mutant library (PDB 1MHP)

## Citation
Clark LA et al. (2006) Affinity enhancement of an in vivo matured therapeutic
antibody using structure-based computational design.
*Protein Sci* 15(5):949-960. doi:10.1110/ps.052030506

## Description
In-silico biophysics-directed mutagenesis of an anti-VEGF-A antibody (PDB 1MHP:
chains H=VH 219aa, L=VL 212aa, A=VEGF-A 192aa). A library of rational mutations
was tested for VEGF binding. Represents a small-scale, curated mutation set.

## Download Instructions

1. Visit the paper: https://doi.org/10.1110/ps.052030506
2. Download supplementary tables (typically in supplementary material).
3. Place raw data in `raw/`.

Note: This paper may only report a small number of variants (10s–100s), not
thousands. If the data is only in a paper table (not machine-readable format),
manual transcription may be needed.

## PDB download
```bash
wget https://files.rcsb.org/download/1MHP.pdb -O raw/1MHP.pdb
```
(Note: 1MHP.pdb is already in DMS_Data/pdbs/ if needed for WT sequences.)

## Status
❌ Not yet downloaded. Run `source_clark.py` after placing raw data in `raw/`.
