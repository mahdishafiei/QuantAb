"""Canonical schema, validation, and shared utilities for the affinity corpus."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CENSORING_THRESHOLD_NM: float = 1000.0  # KD > 1 µM → flag is_censored

SCHEMA: dict[str, pl.DataType] = {
    # Sequences — all required; rows missing any are dropped
    "hc_seq":          pl.Utf8,
    "lc_seq":          pl.Utf8,
    "target_seq":      pl.Utf8,
    # Affinity
    "kd_nm":           pl.Float64,   # KD in nM (positive float)
    "kd_sd_nm":        pl.Float64,   # SD/error in nM (null if unavailable)
    # Provenance
    "source_key":      pl.Utf8,      # Short key e.g. "phillips_2021"
    "source_citation": pl.Utf8,      # Full citation string
    "method":          pl.Utf8,      # SPR | BLI | yeast_display | phage_display | ...
    # Antigen annotation
    "gross_antigen":   pl.Utf8,      # High-level: Flu | VEGF | HER2 | SARS-CoV-2 | ...
    "fine_antigen":    pl.Utf8,      # Specific: H1-HA | H3-HA | RBD | VEGF-A | ...
    # Censoring — keep flagged, never hard-drop
    "is_censored":     pl.Boolean,   # True when KD > 1 µM or hits LoD
    "censor_type":     pl.Utf8,      # "upper_limit" | "lower_limit" | "lod" | null
    # Misc
    "notes":           pl.Utf8,
}

_NON_AA = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clean_seq(s: Optional[str]) -> Optional[str]:
    """Uppercase and strip non-amino-acid characters from a sequence."""
    if s is None or s == "\\" or s.strip() == "":
        return None
    return _NON_AA.sub("", s.upper()) or None


def neg_log_kd_to_nm(neg_log_kd: float) -> float:
    """Convert -log10(KD/M) to KD in nM.

    e.g., neg_log_kd=9.0 → kd_nm=1.0 nM
    """
    return 10.0 ** (9.0 - neg_log_kd)


def apply_censoring(df: pl.DataFrame) -> pl.DataFrame:
    """Set is_censored=True and censor_type='upper_limit' where kd_nm > threshold."""
    return df.with_columns([
        pl.when(pl.col("kd_nm") > CENSORING_THRESHOLD_NM)
          .then(pl.lit(True))
          .otherwise(pl.col("is_censored"))
          .alias("is_censored"),
        pl.when(pl.col("kd_nm") > CENSORING_THRESHOLD_NM)
          .then(pl.lit("upper_limit"))
          .otherwise(pl.col("censor_type"))
          .alias("censor_type"),
    ])


def validate_df(
    df: pl.DataFrame,
    source_key: str = "unknown",
    min_hc: int = 50,
    max_hc: int = 250,
    min_lc: int = 50,
    max_lc: int = 250,
    min_target: int = 30,
) -> pl.DataFrame:
    """Validate a sourced dataframe and drop invalid rows.

    Steps:
      1. Clean sequences (uppercase, strip non-AA)
      2. Drop rows missing hc_seq / lc_seq / target_seq / kd_nm
      3. Drop rows with sequence lengths outside allowed ranges
      4. Drop rows with kd_nm <= 0

    Prints a summary of rows dropped at each step.
    """
    n_in = len(df)

    # Clean sequences
    df = df.with_columns([
        pl.col("hc_seq").map_elements(clean_seq, return_dtype=pl.Utf8),
        pl.col("lc_seq").map_elements(clean_seq, return_dtype=pl.Utf8),
        pl.col("target_seq").map_elements(clean_seq, return_dtype=pl.Utf8),
    ])

    # Drop missing required fields
    df = df.filter(
        pl.col("hc_seq").is_not_null()
        & (pl.col("hc_seq").str.len_chars() > 0)
        & pl.col("lc_seq").is_not_null()
        & (pl.col("lc_seq").str.len_chars() > 0)
        & pl.col("target_seq").is_not_null()
        & (pl.col("target_seq").str.len_chars() > 0)
        & pl.col("kd_nm").is_not_null()
        & (pl.col("kd_nm") > 0)
    )
    n_nonnull = len(df)

    # Length filters
    df = df.filter(
        (pl.col("hc_seq").str.len_chars() >= min_hc)
        & (pl.col("hc_seq").str.len_chars() <= max_hc)
        & (pl.col("lc_seq").str.len_chars() >= min_lc)
        & (pl.col("lc_seq").str.len_chars() <= max_lc)
        & (pl.col("target_seq").str.len_chars() >= min_target)
    )
    n_out = len(df)

    print(
        f"[{source_key}] in={n_in} → nonnull={n_nonnull} "
        f"(dropped {n_in - n_nonnull}) → length-filtered={n_out} "
        f"(dropped {n_nonnull - n_out})"
    )
    return df


def cast_to_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Cast all columns to the canonical SCHEMA dtypes, adding nulls for missing cols."""
    for col, dtype in SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            df = df.with_columns(pl.col(col).cast(dtype))
    return df.select(list(SCHEMA.keys()))


# ---------------------------------------------------------------------------
# PDB sequence extraction utility
# ---------------------------------------------------------------------------

def extract_pdb_chain(pdb_path: Path, chain_id: str) -> str:
    """Extract amino acid sequence from SEQRES records for a given chain."""
    aa_map = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    residues: list[str] = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("SEQRES") and line[11] == chain_id:
                residues.extend(line[19:].split())
    return "".join(aa_map.get(r, "X") for r in residues)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Canonical SCHEMA:")
    for col, dtype in SCHEMA.items():
        print(f"  {col:20s}  {dtype}")

    print("\nCensoring threshold:", CENSORING_THRESHOLD_NM, "nM (1 µM)")
    print("\nclean_seq tests:")
    print("  'EVQLV-es1'  →", clean_seq("EVQLV-es1"))
    print("  None         →", clean_seq(None))
    print("  '\\\\'         →", clean_seq("\\"))
    print("\nneg_log_kd_to_nm tests:")
    print("  9.0 → 1.0 nM :", neg_log_kd_to_nm(9.0))
    print("  7.0 → 100 nM :", neg_log_kd_to_nm(7.0))
    print("\nschema.py self-test passed.")
