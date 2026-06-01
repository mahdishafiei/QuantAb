"""Source script: Dailey et al. 2021 CR9114 single-point mutant KD.

Reads from curated_data/dailey_curated.csv where full HC/LC/target sequences
have already been reconstructed from Kabat-numbered mutation strings
(complex reconstruction requiring ANARCI-based Kabat indexing).

Paper: Dailey et al. (2021) PLACEHOLDER — please update citation once confirmed.
       CR9114 VH single-point mutant scanning vs H1/H3/FluB HA.

Method: yeast display / flow cytometry titeseq (KD from titration curve fit).
Target: Influenza HA (H1, H3, FluB) — sequences from PDB 4FQI and related.
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

SOURCE_KEY = "dailey_2021"
CITATION = (
    "Dailey et al. (2021) Deep mutational scanning of CR9114 VH against "
    "influenza HA subtypes H1, H3, and FluB. [citation to be confirmed]"
)
METHOD = "yeast_display"
GROSS_ANTIGEN = "Flu"

CURATED_CSV = REPO_ROOT / "curated_data" / "dailey_curated.csv"
OUT_FILE = Path(__file__).parent / "dailey_2021.parquet"

# Antigen name → fine antigen label
_ANTIGEN_MAP = {
    "H1-HA": "H1-HA",
    "H3-HA": "H3-HA",
    "fluB-HA": "FluB-HA",
    "FluB-HA": "FluB-HA",
    "GL": "H1-HA",  # germline reverted → tested vs H1
}


def source_dailey() -> pl.DataFrame:
    """Load and annotate the Dailey CR9114 mutant dataset from curated CSV."""
    print(f"\n[{SOURCE_KEY}] Loading from {CURATED_CSV}")

    if not CURATED_CSV.exists():
        raise FileNotFoundError(
            f"curated_data/dailey_curated.csv not found at {CURATED_CSV}. "
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
            pl.col("antigen").map_elements(
                lambda a: _ANTIGEN_MAP.get(a, a), return_dtype=pl.Utf8
            ).alias("fine_antigen"),
            pl.lit(None).cast(pl.Float64).alias("kd_sd_nm"),
            pl.lit(False).alias("is_censored"),
            pl.lit(None).cast(pl.Utf8).alias("censor_type"),
            pl.col("notes").fill_null("CR9114 VH single-point mutant; LC fixed WT"),
        ])
    )

    df = apply_censoring(df)
    df = validate_df(df, source_key=SOURCE_KEY)
    df = cast_to_schema(df)
    return df


if __name__ == "__main__":
    df = source_dailey()
    print(f"\nTotal rows: {len(df):,}")
    print(df.head(3))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_FILE)
    print(f"\nSaved → {OUT_FILE}")
