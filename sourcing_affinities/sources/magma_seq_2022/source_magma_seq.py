"""Source script: MAGMA-seq (4A8 / CC12.1 vs SARS-CoV-2 RBD).

Reads from curated_data/magma_seq_curated.csv where full HC/LC/target
sequences have been pre-reconstructed from mutation-encoded variant strings
in the original SD5 supplementary CSVs.

Papers:
  - Mason DM et al. (2021) Optimization of therapeutic antibodies by predicting
    antigen specificity from antibody sequence via deep learning.
    Nat Biomed Eng 5:600-612. doi:10.1038/s41551-021-00699-9

Method: yeast display titeseq (flow cytometry titration-curve KD estimates).
Target: SARS-CoV-2 RBD — sequence from PDB 6XC3 chain C (231 aa, His-tag stripped).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sourcing_affinities"))
from schema import (
    CENSORING_THRESHOLD_NM,
    apply_censoring,
    cast_to_schema,
    validate_df,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_KEY = "magma_seq_2022"
CITATION = (
    "Mason DM et al. (2021) Optimization of therapeutic antibodies by predicting "
    "antigen specificity from antibody sequence via deep learning. "
    "Nat Biomed Eng 5:600-612. doi:10.1038/s41551-021-00699-9"
)
METHOD = "yeast_display"
GROSS_ANTIGEN = "SARS-CoV-2"
FINE_ANTIGEN = "RBD"

CURATED_CSV = REPO_ROOT / "curated_data" / "magma_seq_curated.csv"
OUT_FILE = Path(__file__).parent / "magma_seq_2022.parquet"


def source_magma_seq() -> pl.DataFrame:
    """Load and annotate the MAGMA-seq dataset from curated CSV."""
    print(f"\n[{SOURCE_KEY}] Loading from {CURATED_CSV}")

    if not CURATED_CSV.exists():
        raise FileNotFoundError(
            f"curated_data/magma_seq_curated.csv not found at {CURATED_CSV}. "
            "Run the original curation pipeline first."
        )

    df = pl.read_csv(CURATED_CSV)
    # Existing columns: hc_seq, lc_seq, target_seq, kd_nm, source, antibody, antigen, notes

    df = (
        df.filter(
            pl.col("hc_seq").is_not_null()
            & pl.col("lc_seq").is_not_null()
            & pl.col("target_seq").is_not_null()
            & pl.col("kd_nm").is_not_null()
            & (pl.col("kd_nm") > 0)
        )
        .with_columns([
            pl.lit(SOURCE_KEY).alias("source_key"),
            pl.lit(CITATION).alias("source_citation"),
            pl.lit(METHOD).alias("method"),
            pl.lit(GROSS_ANTIGEN).alias("gross_antigen"),
            pl.lit(FINE_ANTIGEN).alias("fine_antigen"),
            pl.lit(None).cast(pl.Float64).alias("kd_sd_nm"),
            pl.lit(False).alias("is_censored"),
            pl.lit(None).cast(pl.Utf8).alias("censor_type"),
            pl.col("notes").fill_null(
                "4A8/CC12.1 variant; KD from yeast display titeseq"
            ),
        ])
    )

    df = apply_censoring(df)
    df = validate_df(df, source_key=SOURCE_KEY)
    df = cast_to_schema(df)
    return df


if __name__ == "__main__":
    df = source_magma_seq()
    print(f"\nTotal rows: {len(df):,}")
    print(df.head(3))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_FILE)
    print(f"\nSaved → {OUT_FILE}")
