# Koenig et al. 2017 — Ranibizumab anti-VEGF Fab DMS

## Citation
Koenig P et al. (2017) Mutational landscape of antibody variable domains reveals
a switch between stability and binding optimization at cold temperatures.
*Proc Natl Acad Sci USA* 114(3):497-502. doi:10.1073/pnas.1613231114

## Description
Phage-display DMS of Ranibizumab (anti-VEGF-A Fab) in CDRs + framework regions.
PDB: 2FJG (Ranibizumab Fab bound to VEGF-A fragment).
Chains: H (118 aa VH), L (107 aa VL), V (95 aa VEGF-A fragment).

## Download Instructions

1. Visit the paper page: https://doi.org/10.1073/pnas.1613231114
2. Download Supplementary Dataset (typically CSV/Excel tables in SI)
3. Place raw files in `raw/` subdirectory of this folder.

Alternative: check AbBiBench repository (if/when data is released):
https://github.com/AbBiBench (check for koenig_2017 or 2fjg data)

## Expected raw data format
- A CSV/Excel with columns for: heavy chain mutation (or full sequence),
  light chain mutation (or full sequence), KD or enrichment score.
- May require mutation-to-sequence reconstruction using PDB 2FJG SEQRES.

## Status
❌ Not yet downloaded. Run `source_koenig.py` after placing raw data in `raw/`.
