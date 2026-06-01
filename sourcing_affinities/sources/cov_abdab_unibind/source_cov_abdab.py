"""Source script: CoV-UniBind SPR affinities for SARS-CoV-2 antibodies.

Ingests the quantitative-KD slice of CoV-UniBind (`spr_ab_affinity.parquet`),
which is self-contained: each row already carries paired antibody VH/VL
sequences, the variant-specific antigen sequence, and a measured SPR KD. No join
to CoV-AbDab is needed (CoV-AbDab provides only qualitative binding labels, so
it is not used here).

We ingest ONLY the SPR table (839 KD measurements). CoV-UniBind's other tables
are neutralization IC50 (DRDB), ELISA IC50, and DMS escape scores — different
measurement axes, not equilibrium binding affinity — and are intentionally
skipped (see README).

Units: target_type == "log10_kd" with KD in nM, i.e. target_value = log10(KD/nM).
       → kd_nm = 10 ** target_value. (Values span ~0.001 nM .. 20 µM, median
       ~135 nM — the expected SPR antibody-antigen range.)

Antibody: variable domains antibody_vh_sequence / antibody_vl_sequence (the full
       Fab chains run up to 255 aa and would trip the 250-aa hc_seq ceiling).
Antigen: antigen_sequence (variant-specific RBD or spike ectodomain; 375 unique
       sequences across 73+ lineages — captures the Omicron/Delta/etc. diversity).

Data: HuggingFace InstaDeepAI/cov-unibind, data/spr_ab_affinity.parquet
      (download into raw/; gitignored).
Paper: Bhasin et al. (2025) CoV-UniBind: A Unified Antibody Binding Database for
       SARS-CoV-2. bioRxiv 2025.07.17.665346; Bioinformatics Advances,
       doi:10.1093/bioadv/vbaf328.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sourcing_affinities"))
from schema import (  # noqa: E402
    apply_censoring,
    cast_to_schema,
    validate_df,
)

SOURCE_KEY = "cov_abdab_unibind"
RAW_DIR = Path(__file__).parent / "raw"
SPR_FILE = RAW_DIR / "spr_ab_affinity.parquet"
OUT_FILE = Path(__file__).parent / "cov_abdab_unibind.parquet"

CITATION = (
    "Bhasin S et al. (2025) CoV-UniBind: A Unified Antibody Binding Database for "
    "SARS-CoV-2. bioRxiv 2025.07.17.665346; Bioinformatics Advances "
    "doi:10.1093/bioadv/vbaf328. Antibody sequences originally from CoV-AbDab "
    "(Raybould et al. 2021, doi:10.1093/bioinformatics/btaa739)."
)
METHOD = "SPR"
GROSS_ANTIGEN = "SARS-CoV-2"

# Map CoV-UniBind antigen_domain → our fine_antigen label.
_FINE = {"RBD": "SARS-CoV-2-RBD", "spike": "SARS-CoV-2-Spike"}


def source_cov_abdab() -> pl.DataFrame:
    """Load and annotate the CoV-UniBind SPR KD table."""
    if not SPR_FILE.exists():
        raise FileNotFoundError(
            f"\n[{SOURCE_KEY}] Missing {SPR_FILE}.\n"
            "Download with:\n"
            "  curl -sL https://huggingface.co/datasets/InstaDeepAI/cov-unibind/"
            "resolve/main/data/spr_ab_affinity.parquet -o "
            f"{SPR_FILE}\nSee README.md."
        )

    raw = pl.read_parquet(SPR_FILE)
    print(f"\n[{SOURCE_KEY}] {raw.height:,} SPR rows | "
          f"target_type={raw['target_type'].unique().to_list()}")

    bad_type = raw.filter(pl.col("target_type") != "log10_kd").height
    if bad_type:
        raise ValueError(
            f"{bad_type} rows have target_type != 'log10_kd'; KD conversion "
            "assumes log10(KD/nM). Inspect before ingesting."
        )

    df = raw.with_columns([
        pl.col("antibody_vh_sequence").alias("hc_seq"),
        pl.col("antibody_vl_sequence").alias("lc_seq"),
        pl.col("antigen_sequence").alias("target_seq"),
        # target_value = log10(KD / nM)  →  kd_nm = 10 ** target_value
        (10.0 ** pl.col("target_value")).alias("kd_nm"),
        pl.lit(None, dtype=pl.Float64).alias("kd_sd_nm"),
        pl.lit("kd").alias("measure_type"),
        pl.lit(SOURCE_KEY).alias("source_key"),
        pl.lit(CITATION).alias("source_citation"),
        pl.lit(METHOD).alias("method"),
        pl.lit(GROSS_ANTIGEN).alias("gross_antigen"),
        pl.col("antigen_domain")
          .replace_strict(_FINE, default="SARS-CoV-2-Spike")
          .alias("fine_antigen"),
        pl.lit(False).alias("is_censored"),
        pl.lit(None, dtype=pl.Utf8).alias("censor_type"),
        (pl.lit("CoV-UniBind SPR; ab=") + pl.col("antibody_name")
         + pl.lit("; lineage=") + pl.col("antigen_lineage").fill_null("?")
         + pl.lit("; domain=") + pl.col("antigen_domain")
         + pl.lit("; src=") + pl.col("source_doi").fill_null("?")).alias("notes"),
    ])

    df = apply_censoring(df)
    df = validate_df(df, source_key=SOURCE_KEY)
    df = cast_to_schema(df)
    return df


if __name__ == "__main__":
    df = source_cov_abdab()
    print(f"\nTotal rows: {len(df):,}")
    print(f"  unique antibodies (hc,lc): "
          f"{df.select(['hc_seq', 'lc_seq']).n_unique():,} | "
          f"unique target_seq: {df['target_seq'].n_unique():,}")
    print(f"  censored (KD > 1 µM): {df.filter(pl.col('is_censored')).height:,}")
    print(df.group_by("fine_antigen").len())
    print(df.select(["hc_seq", "target_seq", "kd_nm", "fine_antigen"]).head(3))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_FILE)
    print(f"\nSaved → {OUT_FILE}")
