"""Build the unified, normalized antibody-antigen affinity dataset.

Merges the curated DMS data (``filtered_data.csv``) with the usable subset of Terrence's
ANDD collection, harmonizes everything to pKd (``-log10 KD[M]``), drops degenerate groups,
and z-scores the label within each protocol/antigen group so a single regression can run
across all of it.

Only rows usable for supervised regression are kept: an antibody **heavy** chain, an
**antigen** sequence, and a valid affinity label. (Light chain is optional — nanobodies and
heavy-only records are kept with an empty light chain.)

Usage
-----
python scripts/build_unified_affinity.py \
    --curated curated_data/filtered_data.csv \
    --andd    curated_data/ANDD_dataset.xlsx \
    --out     curated_data/unified_affinity.csv
"""

import argparse
from pathlib import Path

import polars as pl

from quantab.data import load_curated, load_andd, normalize_affinity

# Phillips CR9114 vs H3-HA is ~88% pinned at the assay ceiling (degenerate) — drop it.
DROP_GROUPS = {("phillips2021", "H3-HA")}

# ESM-2 usable context (1024). Antigens longer than this are truncated when embedded.
ANTIGEN_MAX_LEN = 1024

COLS = ["heavy", "light", "target", "affinity", "source", "antigen", "dataset"]


def _prep(df: pl.DataFrame) -> pl.DataFrame:
    if "light" not in df.columns:
        df = df.with_columns(pl.lit("").alias("light"))
    return df.with_columns(pl.col("light").fill_null("")).select(COLS)


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    parts = []

    print(f"Loading curated DMS data: {args.curated}")
    dms = load_curated(Path(args.curated))
    # drop degenerate (heavily censored) groups
    for src, ag in DROP_GROUPS:
        before = len(dms)
        dms = dms.filter(~((pl.col("source") == src) & (pl.col("antigen") == ag)))
        print(f"  dropped {before - len(dms)} rows for degenerate group {src}/{ag}")
    parts.append(_prep(dms))

    if args.andd and Path(args.andd).exists():
        print(f"Loading ANDD usable subset: {args.andd}")
        andd = load_andd(Path(args.andd))
        print(f"  ANDD usable rows: {len(andd)}  ({andd['antigen'].n_unique()} antigens)")
        parts.append(_prep(andd))
    else:
        print("ANDD file not provided/found — building from curated DMS only.")

    df = pl.concat(parts, how="vertical")

    # normalization group: per (source, antigen) for the DMS protocols; ANDD as one
    # block (its pKd is already an absolute molar scale across sub-sources).
    df = df.with_columns(
        pl.when(pl.col("dataset") == "ANDD")
        .then(pl.lit("ANDD"))
        .otherwise(pl.col("source") + pl.lit("|") + pl.col("antigen"))
        .alias("norm_group")
    )

    # clean + dedup on the sequence triple
    df = (
        df.with_columns(pl.col("heavy").str.strip_chars(), pl.col("target").str.strip_chars())
        .unique(subset=["heavy", "light", "target"], keep="first")
        .drop_nulls(subset=["heavy", "target", "affinity"])
    )

    # z-score the affinity within each normalization group
    df = normalize_affinity(df, by=("norm_group",), method="zscore")

    # sequence-size metadata so downstream knows what will/won't fit the model context.
    # Antibody (heavy[+light]) fits in 512; antigens can exceed it — flag truncation.
    df = df.with_columns(
        pl.col("heavy").str.len_chars().alias("heavy_len"),
        pl.col("light").str.len_chars().alias("light_len"),
        pl.col("target").str.len_chars().alias("target_len"),
    ).with_columns(
        (pl.col("target_len") > ANTIGEN_MAX_LEN).alias("target_truncated")
    )

    df = df.select(
        ["heavy", "light", "target", "affinity", "affinity_norm",
         "source", "antigen", "norm_group", "dataset",
         "heavy_len", "light_len", "target_len", "target_truncated"]
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)

    print(f"\nUnified dataset → {out}  ({len(df)} rows)")
    print("\nrows per dataset:")
    print(df.group_by("dataset").agg(pl.len().alias("n")).sort("n", descending=True))
    print("\nrows per normalization group (top 12):")
    print(df.group_by("norm_group").agg(pl.len().alias("n")).sort("n", descending=True).head(12))
    print(f"\nunique antigens: {df['antigen'].n_unique()} | unique heavy: {df['heavy'].n_unique()}")
    print(f"affinity (pKd) range: {df['affinity'].min():.2f} .. {df['affinity'].max():.2f}")
    print(f"affinity_norm mean/std: {df['affinity_norm'].mean():.3f} / {df['affinity_norm'].std():.3f}")
    n_paired = df.filter(pl.col('light').str.len_chars() > 0).height
    print(f"rows with a light chain (paired): {n_paired} / {len(df)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the unified normalized affinity dataset.")
    p.add_argument("--curated", required=True, help="Path to filtered_data.csv")
    p.add_argument("--andd", default=None, help="Path to ANDD_dataset.xlsx (optional)")
    p.add_argument("--out", default="curated_data/unified_affinity.csv", help="Output CSV path")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
