# Minot & Reddy 2024 — Anti-Ang2 Antibody 5A12 Yeast Display DMS

## Citation
Minot M & Reddy ST (2024) Deep mutational scanning of antibody 5A12 against
Angiopoietin-2. [confirm full citation — check bioRxiv or publication]

## Description
Yeast display DMS of antibody 5A12 against Angiopoietin-2 (Ang2).
AbBiBench ID: 5a12_ang2. CDR mutations measured by yeast display enrichment.
PDB: no specific structure listed; antibody is 5A12.

## Download Instructions

1. Search for "Minot Reddy 2024 antibody DMS Angiopoietin" on bioRxiv or PubMed.
2. Check Zenodo for associated dataset (DOI in paper).
3. Check GitHub for: github.com/srreddy or similar author page.
4. Place raw data CSV/Excel in `raw/`.

## Expected data format
- Variant library table with: HC/LC mutation or full sequence, KD or enrichment
  score, replicate measurements.
- May need WT 5A12 sequence reconstruction from the paper.

## Antigen sequence
Angiopoietin-2 (Ang2): UniProt O15123 (human). The binding domain is the
fibrinogen-like domain (FLD, ~240 aa, positions 284-496 in full-length Ang2).
Full-length Ang2 is 496 aa; retrieve from UniProt O15123.

## Status
❌ Not yet downloaded. Run `source_minot.py` after placing raw data in `raw/`.
