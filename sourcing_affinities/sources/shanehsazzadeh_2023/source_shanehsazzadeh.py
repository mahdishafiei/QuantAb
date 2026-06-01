"""Source script: Shanehsazzadeh et al. 2023 trastuzumab HCDR3 zero-shot binders.

SPR-measured KD (nM) for 422 AI-generated antibodies with varying HCDR3s
grafted into the trastuzumab VH/VL scaffold.

Strategy: each entry has only an HCDR3 sequence. We reconstruct the full VH by
replacing the trastuzumab WT HCDR3 in the PDB 1N8Z scaffold. LC and HER2
antigen are fixed from PDB 1N8Z.

Paper: Shanehsazzadeh A et al. (2023) Unlocking de novo antibody design with
       generative artificial intelligence. bioRxiv. doi:10.1101/2023.01.08.523187

Method: SPR (surface plasmon resonance), direct KD measurement.
Target: HER2 (ErbB2), ECD — PDB 1N8Z chain C (607 aa).
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
    extract_pdb_chain,
    validate_df,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_KEY = "shanehsazzadeh_2023"
CITATION = (
    "Shanehsazzadeh A et al. (2023) Unlocking de novo antibody design with "
    "generative artificial intelligence. bioRxiv. doi:10.1101/2023.01.08.523187"
)
METHOD = "SPR"
GROSS_ANTIGEN = "HER2"
FINE_ANTIGEN = "HER2-ECD"

RAW_DIR = (
    REPO_ROOT
    / "DMS_Data"
    / "06_shanehsazzadeh_trastuzumab"
    / "unlocking-de-novo-antibody-design"
)
PDB_DIR = REPO_ROOT / "DMS_Data" / "pdbs"
OUT_FILE = Path(__file__).parent / "shanehsazzadeh_2023.parquet"

# Trastuzumab HCDR3 wild-type (from PDB 1N8Z chain B, Kabat HCDR3)
TRASTUZUMAB_HCDR3_WT = "SRWGGDGFYAMDY"


def _load_scaffold() -> tuple[str, str, str]:
    """Load trastuzumab VH, VL, and HER2 sequences from PDB 1N8Z.

    Returns:
        vh: trastuzumab VH+CH1 Fab sequence (220 aa)
        vl: trastuzumab VL+CL Fab sequence (214 aa)
        her2: HER2 ECD sequence (607 aa)
    """
    vh = extract_pdb_chain(PDB_DIR / "1N8Z.pdb", "B")   # 220 aa
    vl = extract_pdb_chain(PDB_DIR / "1N8Z.pdb", "A")   # 214 aa
    her2 = extract_pdb_chain(PDB_DIR / "1N8Z.pdb", "C") # 607 aa
    assert TRASTUZUMAB_HCDR3_WT in vh, (
        f"WT HCDR3 '{TRASTUZUMAB_HCDR3_WT}' not found in scaffold VH. "
        "Check PDB 1N8Z chain B SEQRES."
    )
    return vh, vl, her2


def _graft_hcdr3(vh_scaffold: str, hcdr3: str) -> str:
    """Replace WT HCDR3 in the trastuzumab scaffold with a variant HCDR3."""
    if not hcdr3 or len(hcdr3) < 3:
        return ""
    return vh_scaffold.replace(TRASTUZUMAB_HCDR3_WT, hcdr3)


def source_shanehsazzadeh() -> pl.DataFrame:
    """Load and annotate the Shanehsazzadeh 2023 zero-shot binders dataset."""
    print(f"\n[{SOURCE_KEY}] Loading from {RAW_DIR}")
    vh_scaffold, vl_fixed, her2_seq = _load_scaffold()
    print(
        f"  Scaffold — VH: {len(vh_scaffold)} aa, VL: {len(vl_fixed)} aa, "
        f"HER2: {len(her2_seq)} aa"
    )
    print(f"  WT HCDR3: {TRASTUZUMAB_HCDR3_WT}")

    csv_file = RAW_DIR / "zero-shot-binders.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"Expected {csv_file}")

    df = pl.read_csv(csv_file)
    # columns: HCDR3, KD (nM), -log(KD (M)), HCDR3 Edit Distance to Trastuzumab, ...

    df = df.filter(
        pl.col("HCDR3").is_not_null() & pl.col("KD (nM)").is_not_null()
    ).with_columns([
        pl.col("HCDR3").map_elements(
            lambda h: _graft_hcdr3(vh_scaffold, h), return_dtype=pl.Utf8
        ).alias("hc_seq"),
        pl.lit(vl_fixed).alias("lc_seq"),
        pl.lit(her2_seq).alias("target_seq"),
        pl.col("KD (nM)").alias("kd_nm"),
        pl.lit(None).cast(pl.Float64).alias("kd_sd_nm"),
        pl.lit(SOURCE_KEY).alias("source_key"),
        pl.lit(CITATION).alias("source_citation"),
        pl.lit(METHOD).alias("method"),
        pl.lit(GROSS_ANTIGEN).alias("gross_antigen"),
        pl.lit(FINE_ANTIGEN).alias("fine_antigen"),
        pl.lit(False).alias("is_censored"),
        pl.lit(None).cast(pl.Utf8).alias("censor_type"),
        pl.col("HCDR3").map_elements(
            lambda h: f"HCDR3={h}; VH grafted into trastuzumab scaffold (PDB 1N8Z)",
            return_dtype=pl.Utf8,
        ).alias("notes"),
    ])

    # Drop rows where grafting failed (empty or short VH)
    df = df.filter(
        pl.col("hc_seq").is_not_null()
        & (pl.col("hc_seq").str.len_chars() > 50)
    )

    df = apply_censoring(df)
    df = validate_df(df, source_key=SOURCE_KEY)
    df = cast_to_schema(df)
    return df


if __name__ == "__main__":
    df = source_shanehsazzadeh()
    print(f"\nTotal rows: {len(df):,}")
    print(df.head(3))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_FILE)
    print(f"\nSaved → {OUT_FILE}")
