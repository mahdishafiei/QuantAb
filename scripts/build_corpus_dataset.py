"""Build the canonical training dataset (unified_affinity_v3) from the affinity corpus.

Takes the team's affinity-sourcing output (`affinity_corpus_filtered.parquet`) — which aggregates
Phillips, Engelhart/AlphaSeq, Dailey, ANDD, Shanehsazzadeh, MAGMA-seq, etc. — and produces a
clean, normalized, regression-ready table:

- keep only **measured** methods (drop `Predict`) and **physically sane KD** (1 mM .. 0.1 pM);
- require an antibody heavy chain + an antigen sequence; de-dup on (heavy, light, target);
- **z-score the affinity within each (method, antigen) group** -> `affinity_norm` (the target);
- add `dataset` (= source), sequence-length columns, and a `target_truncated` flag.

Usage
-----
python scripts/build_corpus_dataset.py \
    --corpus affinity_corpus/affinity_corpus_filtered.parquet \
    --out    curated_data/unified_affinity_v3.csv
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from quantab.data import normalize_affinity

KD_LO, KD_HI = 1e-12, 1e-3          # sane KD bounds in molar (0.1 pM .. 1 mM) -> pKd 12..3
ANTIGEN_MAX_LEN = 1024              # ESM-2 context; longer antigens get truncated when embedded


def run(args: argparse.Namespace) -> None:
    c = pl.read_parquet(args.corpus)
    print(f"corpus rows: {len(c)}")

    c = c.with_columns([
        pl.col("hc_seq").str.strip_chars().alias("heavy"),
        pl.col("lc_seq").fill_null("").str.strip_chars().alias("light"),
        pl.col("target_seq").str.strip_chars().alias("target"),
    ])
    clean = (
        c.filter(pl.col("method") != "Predict")
         .filter((pl.col("affinity") >= 3) & (pl.col("affinity") <= 13))
         .filter(pl.col("heavy").str.len_chars() >= 50)
         .filter(pl.col("target").str.len_chars() >= 20)
         .rename({"source_key": "source", "fine_antigen": "antigen"})
    )
    print(f"after clean (measured + sane KD + seqs): {len(clean)}")

    clean = clean.with_columns(
        (pl.col("method") + pl.lit("|") + pl.col("antigen")).alias("norm_group"),
        pl.col("source").alias("dataset"),
    )
    clean = clean.unique(subset=["heavy", "light", "target"], keep="first")
    clean = normalize_affinity(clean, by=("norm_group",), method="zscore")
    clean = clean.with_columns([
        pl.col("heavy").str.len_chars().alias("heavy_len"),
        pl.col("light").str.len_chars().alias("light_len"),
        pl.col("target").str.len_chars().alias("target_len"),
    ]).with_columns((pl.col("target_len") > ANTIGEN_MAX_LEN).alias("target_truncated"))

    out = clean.select([
        "heavy", "light", "target", "affinity", "affinity_norm",
        "source", "antigen", "method", "norm_group", "gross_antigen", "dataset",
        "kd_nm", "kd_sd_nm", "n_sources_confirmed",
        "heavy_len", "light_len", "target_len", "target_truncated",
    ])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out)

    print(f"\nunified_affinity_v3 -> {args.out}  ({len(out)} rows)")
    print(f"unique antigens: {out['antigen'].n_unique()} | unique heavy: {out['heavy'].n_unique()}")
    print(f"affinity (pKd): {out['affinity'].min():.1f} .. {out['affinity'].max():.1f} | "
          f"affinity_norm mean/std: {out['affinity_norm'].mean():.3f}/{out['affinity_norm'].std():.3f}")
    print(out.group_by("dataset").agg(pl.len().alias("n")).sort("n", descending=True))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified_affinity_v3 from the affinity corpus.")
    p.add_argument("--corpus", default="affinity_corpus/affinity_corpus_filtered.parquet")
    p.add_argument("--out", default="curated_data/unified_affinity_v3.csv")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
