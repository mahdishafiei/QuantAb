"""Source script: Engelhart et al. 2022 AAYL AlphaSeq anti-SARS-CoV-2 DMS.

Re-processes raw CSV files from DMS_Data/05_engelhart_aayl/.
Full HC+LC sequences available in each CSV. The binding_score column
represents -log10(KD/M) estimated by the AlphaSeq pipeline
(Bayesian enrichment model, not direct SPR/BLI measurement).

Paper: Engelhart E et al. (2022) A dataset comprised of binding interactions
       for 104,972 antibodies against a SARS-CoV-2 peptide. Sci Data / Protein
       Eng Des Sel 35:gzac009. doi:10.1093/protein/gzac009

Note on units: binding_score ≈ -log10(KD/M) (pseudo-KD from AlphaSeq).
       kd_nm = 10^(9 - binding_score)
       Method flagged as "phage_display_alphaSeq" to distinguish from direct
       biophysical measurements.

Target antigen: SARS-CoV-2 Spike heptad-repeat-2 (HR2) target peptide
       PDVDLGDISGINAS (14 aa) — the AlphaSeq target the AAYL antibodies were
       selected against (confirmed by AbBiBench AAYL49_bca.pdb chain A;
       "DISGINAS" is the canonical start of Spike HR2).
       NOTE (QC fix 2026-06-01): earlier versions of this script mislabeled the
       antigen as VEGF-A and used PDB 1MHP chain A — which is actually the
       integrin-α1 I-domain, not VEGF. Both the label and the sequence were wrong.
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

SOURCE_KEY = "engelhart_2022"
CITATION = (
    "Engelhart E et al. (2022) A dataset comprised of binding interactions for "
    "104,972 antibodies against a SARS-CoV-2 peptide. "
    "Protein Eng Des Sel 35:gzac009. doi:10.1093/protein/gzac009"
)
METHOD = "phage_display_alphaSeq"
GROSS_ANTIGEN = "SARS-CoV-2"
FINE_ANTIGEN = "SARS-CoV-2-Spike-HR2"

# SARS-CoV-2 Spike HR2 target peptide the AAYL antibodies were raised against.
SPIKE_HR2_PEPTIDE = "PDVDLGDISGINAS"

RAW_DIR = REPO_ROOT / "DMS_Data" / "05_engelhart_aayl"
OUT_FILE = Path(__file__).parent / "engelhart_2022.parquet"

# Files to process (exclude files without binding_score or light chain)
TARGET_FILES = [
    "aayl49_benchmarking_data.csv",
    "aayl51_benchmarking_data.csv",
    "aayl49_ML_benchmarking_data.csv",
    "aayl50_LC_benchmarking_data.csv",
    "aayl52_LC_benchmarking_data.csv",
]


def source_engelhart() -> pl.DataFrame:
    """Load, convert, and annotate the Engelhart 2022 AAYL dataset."""
    print(f"\n[{SOURCE_KEY}] Loading from {RAW_DIR}")
    target_seq = SPIKE_HR2_PEPTIDE
    print(f"  SARS-CoV-2 Spike HR2 target peptide: {target_seq} ({len(target_seq)} aa)")

    parts: list[pl.DataFrame] = []
    for csv_name in TARGET_FILES:
        csv_file = RAW_DIR / csv_name
        if not csv_file.exists():
            print(f"  Skipping {csv_name}: file not found.")
            continue

        df = pl.read_csv(csv_file)

        # Must have heavy_chain_seq and binding_score
        if "heavy_chain_seq" not in df.columns or "binding_score" not in df.columns:
            print(f"  Skipping {csv_name}: missing required columns.")
            continue

        # Light chain may be absent for LC-only files
        if "light_chain_seq" not in df.columns:
            print(f"  Skipping {csv_name}: no light_chain_seq column.")
            continue

        df = (
            df.filter(
                pl.col("heavy_chain_seq").is_not_null()
                & pl.col("light_chain_seq").is_not_null()
                & pl.col("binding_score").is_not_null()
            )
            .with_columns([
                pl.col("heavy_chain_seq").alias("hc_seq"),
                pl.col("light_chain_seq").alias("lc_seq"),
                pl.lit(target_seq).alias("target_seq"),
                # binding_score = -log10(KD/M) → kd_nm = 10^(9 - binding_score)
                (10.0 ** (9.0 - pl.col("binding_score"))).alias("kd_nm"),
                pl.lit(None).cast(pl.Float64).alias("kd_sd_nm"),
                pl.lit(SOURCE_KEY).alias("source_key"),
                pl.lit(CITATION).alias("source_citation"),
                pl.lit(METHOD).alias("method"),
                pl.lit(GROSS_ANTIGEN).alias("gross_antigen"),
                pl.lit(FINE_ANTIGEN).alias("fine_antigen"),
                pl.lit(False).alias("is_censored"),
                pl.lit(None).cast(pl.Utf8).alias("censor_type"),
                pl.lit(
                    f"file={csv_name}; pseudo-KD from AlphaSeq phage-display pipeline"
                ).alias("notes"),
            ])
            .select([
                "hc_seq", "lc_seq", "target_seq", "kd_nm", "kd_sd_nm",
                "source_key", "source_citation", "method",
                "gross_antigen", "fine_antigen",
                "is_censored", "censor_type", "notes",
            ])
        )

        df = apply_censoring(df)
        n = len(df)
        print(f"  {csv_name}: {n:>6,} rows")
        parts.append(df)

    if not parts:
        raise RuntimeError(f"No data loaded from {RAW_DIR}")

    combined = pl.concat(parts)
    # min_target=10: the SARS-CoV-2 HR2 target is a 14-aa peptide, below the
    # default 30-residue floor meant to drop fragment/garbage antigens.
    combined = validate_df(combined, source_key=SOURCE_KEY, min_target=10)
    combined = cast_to_schema(combined)
    return combined


if __name__ == "__main__":
    df = source_engelhart()
    print(f"\nTotal rows: {len(df):,}")
    print(df.head(3))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_FILE)
    print(f"\nSaved → {OUT_FILE}")
