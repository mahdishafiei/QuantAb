# CoV-UniBind — SARS-CoV-2 antibody SPR affinities

## Status: ✅ INGESTED (839 SPR KD rows)

Implemented from **CoV-UniBind** alone. CoV-AbDab is **not** used: CoV-UniBind's
SPR table is self-contained (paired VH/VL + antigen sequence + KD per row), and
CoV-AbDab only carries qualitative binding labels.

## Citations
- CoV-UniBind: Bhasin S et al. (2025) CoV-UniBind: A Unified Antibody Binding
  Database for SARS-CoV-2. bioRxiv 2025.07.17.665346; Bioinformatics Advances
  doi:10.1093/bioadv/vbaf328.
- Antibody sequences originate from CoV-AbDab (Raybould MIJ et al. 2021,
  Bioinformatics 37(5):734-735, doi:10.1093/bioinformatics/btaa739).

## Data download
```bash
curl -sL https://huggingface.co/datasets/InstaDeepAI/cov-unibind/resolve/main/data/spr_ab_affinity.parquet \
  -o raw/spr_ab_affinity.parquet
```
(`raw/` is gitignored.) CoV-UniBind lives at
https://huggingface.co/datasets/InstaDeepAI/cov-unibind

## What we ingest
Only `spr_ab_affinity.parquet` (839 rows) — the quantitative equilibrium-KD
slice. CoV-UniBind's other tables are skipped on purpose:
- `drdb_binding_potency` (neutralization IC50), `jian_elisa_ab_ic50` (ELISA
  IC50) — IC50 is not equilibrium binding affinity.
- `dms_*_escape` — DMS escape scores, a different axis.

## Mapping to schema
- `hc_seq` / `lc_seq` ← `antibody_vh_sequence` / `antibody_vl_sequence` (variable
  domains; full Fab chains reach 255 aa and trip the 250-aa ceiling).
- `target_seq` ← `antigen_sequence` (variant-specific RBD or spike ectodomain;
  375 unique sequences across 73+ lineages — captures Omicron/Delta/etc.).
- `kd_nm` ← `10 ** target_value` (target_type is `log10_kd` with KD in nM;
  range ~0.001 nM .. 20 µM, median ~135 nM). `measure_type="kd"`.
- `fine_antigen` ← `SARS-CoV-2-RBD` (514) or `SARS-CoV-2-Spike` (325).
- method = SPR. 296/839 are weak binders (KD > 1 µM) → flagged is_censored.
