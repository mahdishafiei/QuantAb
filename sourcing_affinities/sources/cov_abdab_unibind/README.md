# CoV-AbDab + CoV-UniBind — SARS-CoV-2 Antibodies with Quantitative KD

## Citations
- CoV-AbDab: Raybould MIJ et al. (2021) CoV-AbDab: the coronavirus antibody
  database. Bioinformatics 37(5):734-735. doi:10.1093/bioinformatics/btaa739

- CoV-UniBind: [bioRxiv 2025.07.17.665346v1] — Quantitative binding database
  pairing CoV-AbDab antibody sequences with measured KD values.
  doi:10.1101/2025.07.17.665346

## Description
CoV-AbDab provides paired HC+LC sequences for SARS-CoV-2 antibodies with
known neutralization and structural data. CoV-UniBind extends this with
quantitative KD values (SPR/BLI) from the literature. Together they provide
the most comprehensive paired-sequence + affinity data for COVID-19 antibodies.

## Download Instructions

### 1. CoV-AbDab sequences
```bash
# Download the full CoV-AbDab CSV (updated monthly)
wget "https://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/CoV-AbDab_260524.csv" \
  -O raw/CoV-AbDab.csv
# Or use the latest download link from: https://opig.stats.ox.ac.uk/webapps/covabdab/
```

### 2. CoV-UniBind KD data
```bash
# From the bioRxiv preprint supplementary material
# doi: 10.1101/2025.07.17.665346
# Download the quantitative binding affinity table from the paper supplement
# Place as raw/CoV-UniBind_KD.csv (or .xlsx)
```

## Expected columns from CoV-AbDab
`Ab or Nb`, `Heavy Variable Sequence`, `Light Variable Sequence`,
`Protein + Epitope`, `Neutralising Vs`, `PDB`, ...

## Expected columns from CoV-UniBind
Antibody ID (to join with CoV-AbDab), antigen/strain info, KD (nM),
measurement method (SPR/BLI), publication reference.

## Notes
- KD values vs IC50/EC50: CoV-UniBind should provide KD specifically.
  Ignore neutralization IC50 values — these are not binding affinities.
- Many antibodies bind RBD; some bind NTD, S2, or other spike regions.
  Capture fine_antigen (RBD | NTD | S2 | spike-full) carefully.
- Both SARS-CoV-2 and SARS-CoV-1 antibodies may be present — track species.
- Clustering is critical: many antibodies from convergent clones.

## Status
❌ Not yet downloaded. Run `source_cov_abdab.py` after placing raw data in `raw/`.
