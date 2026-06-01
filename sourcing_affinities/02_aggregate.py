"""Pipeline step 2: Aggregate, filter, dedup, and produce the affinity corpus.

Reads all available *.parquet files from sources/*/,
merges them, applies cross-source deduplication, computes affinity
(-log10(KD/M)), and writes:
  - outputs/affinity_corpus.parquet        (all rows, censored flagged)
  - outputs/affinity_corpus_filtered.parquet (is_censored=False only)

Usage:
    python sourcing_affinities/02_aggregate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

SOURCING_DIR = Path(__file__).parent
SOURCES_DIR = SOURCING_DIR / "sources"
OUT_DIR = SOURCING_DIR / "outputs"
sys.path.insert(0, str(SOURCING_DIR))

from schema import SCHEMA  # noqa: E402

CORPUS_PATH = OUT_DIR / "affinity_corpus.parquet"
FILTERED_PATH = OUT_DIR / "affinity_corpus_filtered.parquet"


def load_all_sources() -> pl.DataFrame:
    """Load all available source parquets."""
    parts = []
    for parquet_path in sorted(SOURCES_DIR.glob("*/*.parquet")):
        try:
            df = pl.read_parquet(parquet_path)
            # Ensure all schema columns present
            for col, dtype in SCHEMA.items():
                if col not in df.columns:
                    df = df.with_columns(pl.lit(None).cast(dtype).alias(col))
            df = df.select(list(SCHEMA.keys()))
            parts.append(df)
            print(f"  Loaded {parquet_path.parent.name}: {len(df):,} rows")
        except Exception as e:
            print(f"  ERROR loading {parquet_path}: {e}")
    if not parts:
        raise RuntimeError("No source parquets found. Run 01_clean_sources.py first.")
    return pl.concat(parts)


def dedup_cross_source(df: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate identical (hc_seq, lc_seq, target_seq) tuples across sources.

    When the same antibody-antigen pair appears in multiple sources:
    - Count how many sources confirmed it (n_sources_confirmed)
    - Keep the row with the lowest kd_nm (best / tightest measured affinity)
    """
    key = ["hc_seq", "lc_seq", "target_seq"]

    # Count confirmations per unique triple
    confirmation_counts = (
        df.group_by(key)
        .agg(pl.len().alias("n_sources_confirmed"))
    )

    # Within each (hc_seq, lc_seq, target_seq) group, keep the best measurement:
    # lowest kd_nm for KD assays, else highest binding_score (higher = stronger).
    df_ranked = (
        df.sort(
            ["kd_nm", "binding_score"],
            descending=[False, True],
            nulls_last=True,
        )
        .with_columns(
            pl.int_range(pl.len()).over(key).alias("_rank")
        )
        .filter(pl.col("_rank") == 0)
        .drop("_rank")
    )

    df_out = df_ranked.join(confirmation_counts, on=key, how="left")
    return df_out


def compute_affinity(df: pl.DataFrame) -> pl.DataFrame:
    """Add affinity = -log10(KD/M); backfill binding_score and measure_type.

    - ``affinity`` is the absolute pKD; null wherever kd_nm is null (non-KD assays).
    - ``measure_type`` defaults to "kd" for legacy sources that predate the field.
    - ``binding_score`` (monotonic, higher = stronger) is backfilled from affinity
      for KD rows so every row carries a rank-usable score.
    """
    df = df.with_columns(
        (-pl.col("kd_nm").log(10) + 9.0).alias("affinity")
    )
    return df.with_columns([
        pl.col("measure_type").fill_null("kd"),
        pl.coalesce(["binding_score", "affinity"]).alias("binding_score"),
    ])


def main() -> None:
    print("\n[02_aggregate] Loading source parquets...")
    df = load_all_sources()
    n_total = len(df)
    print(f"\nTotal rows loaded: {n_total:,}")

    # Per-source breakdown
    print("\nRows per source:")
    breakdown = df.group_by("source_key").len().sort("len", descending=True)
    print(breakdown)

    # Cross-source dedup
    print("\nDeduplicating across sources...")
    df = dedup_cross_source(df)
    n_dedup = len(df)
    print(f"After dedup: {n_dedup:,} rows (removed {n_total - n_dedup:,} duplicates)")

    # Add affinity column
    df = compute_affinity(df)

    # Censoring summary
    n_censored = df.filter(pl.col("is_censored")).height
    n_filtered = n_dedup - n_censored
    print(f"\nCensored (KD > 1 µM or LoD): {n_censored:,}")
    print(f"Non-censored (usable KD):     {n_filtered:,}")

    # Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(CORPUS_PATH)
    print(f"\nSaved full corpus  → {CORPUS_PATH}  ({n_dedup:,} rows)")

    df_filtered = df.filter(~pl.col("is_censored"))
    df_filtered.write_parquet(FILTERED_PATH)
    print(f"Saved filtered     → {FILTERED_PATH}  ({len(df_filtered):,} rows)")

    # Summary by gross antigen
    print("\nFiltered rows by gross antigen:")
    ag_summary = (
        df_filtered.group_by(["gross_antigen", "fine_antigen"])
        .len()
        .sort("len", descending=True)
    )
    print(ag_summary.head(30))

    # Summary by method
    print("\nFiltered rows by method:")
    method_summary = df_filtered.group_by("method").len().sort("len", descending=True)
    print(method_summary)


if __name__ == "__main__":
    main()
