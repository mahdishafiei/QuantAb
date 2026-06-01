# Warszawski et al. 2020 — Yeast display DMS (PDB 1MLC)

## Citation
Warszawski S et al. (2020) Optimizing antibody affinity and stability by the
automated design of the variable light-heavy chain interfaces.
*PLOS Comput Biol* 16(4):e1007828. doi:10.1371/journal.pcbi.1007828

## Description
Yeast display DMS of an anti-VEGF Fab (PDB 1MLC: chains B, A for VH, VL;
chain E for VEGF-A). CDR and interface mutations, binding to VEGF measured
by yeast display / flow cytometry.

## Download Instructions

1. Download supplementary data from: https://doi.org/10.1371/journal.pcbi.1007828
2. Supplementary files are typically hosted as S1-SX Data on PLOS website.
3. Place raw CSV/Excel in `raw/`.

Alternative: AbBiBench reference lists this as PDB 1mlc — may be on GitHub
if AbBiBench ever releases processed data.

## Expected raw data format
Mutation table with columns for: variant ID, mutation string (position + AA),
binding score or KD, selection/enrichment. WT sequences from PDB 1MLC.

## Status
❌ Not yet downloaded. Run `source_warszawski.py` after placing raw data in `raw/`.
