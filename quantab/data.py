"""Dataset loaders — each returns a polars DataFrame with columns: heavy, light (optional), affinity."""

from pathlib import Path
from typing import Optional
import polars as pl


def load_phillips(data_dir: Path) -> pl.DataFrame:
    """Load Phillips et al. 2021 CR9114 DMS dataset.

    Returns columns: heavy, light, affinity, dataset.
    """
    dfs = []
    csv_dir = data_dir / "02_phillips_cr9114_cr6261"
    for csv_file in sorted(csv_dir.glob("*.csv")):
        df = pl.read_csv(csv_file)
        # columns: genotype, heavy, light, h3_mean, h9_mean_neg_log_kd, fitness
        # affinity column name varies by file — find the neg_log_kd column
        aff_col = next(c for c in df.columns if "neg_log_kd" in c)
        df = (
            df.rename({aff_col: "affinity"})
            .select(["heavy", "light", "affinity"])
            .with_columns(pl.lit(csv_file.stem).alias("dataset"))
            .drop_nulls(subset=["heavy", "affinity"])
        )
        dfs.append(df)
    return pl.concat(dfs)


def load_engelhart(data_dir: Path) -> pl.DataFrame:
    """Load Engelhart et al. AAYL benchmarking datasets.

    Columns: heavy_chain_seq, binding_score, light_chain_seq.
    Returns columns: heavy, light, affinity, dataset.
    """
    dfs = []
    eng_dir = data_dir / "05_engelhart_aayl"
    for csv_file in sorted(eng_dir.glob("*benchmarking_data.csv")):
        df = pl.read_csv(csv_file)
        if "heavy_chain_seq" not in df.columns or "binding_score" not in df.columns:
            continue
        df = (
            df.rename({"heavy_chain_seq": "heavy", "binding_score": "affinity"})
            .with_columns(
                pl.col("light_chain_seq").alias("light")
                if "light_chain_seq" in df.columns
                else pl.lit(None).cast(pl.Utf8).alias("light")
            )
            .select(["heavy", "light", "affinity"])
            .with_columns(pl.lit(csv_file.stem).alias("dataset"))
            .drop_nulls(subset=["heavy", "affinity"])
        )
        dfs.append(df)
    return pl.concat(dfs) if dfs else pl.DataFrame({"heavy": [], "light": [], "affinity": [], "dataset": []})


def load_trastuzumab(data_dir: Path) -> pl.DataFrame:
    """Load Shanehsazzadeh trastuzumab HCDR3 variants.

    Returns columns: heavy (HCDR3 sequence used as proxy), affinity, dataset.
    """
    csv_file = (
        data_dir
        / "06_shanehsazzadeh_trastuzumab"
        / "unlocking-de-novo-antibody-design"
        / "zero-shot-binders.csv"
    )
    if not csv_file.exists():
        return pl.DataFrame({"heavy": [], "affinity": [], "dataset": []})
    df = pl.read_csv(csv_file)
    # columns: HCDR3, KD (nM), -log(KD (M)), ...
    df = (
        df.rename({"HCDR3": "heavy", "-log(KD (M))": "affinity"})
        .select(["heavy", "affinity"])
        .with_columns(pl.lit("trastuzumab_zero_shot").alias("dataset"))
        .drop_nulls()
    )
    return df


def load_all(data_dir: Path) -> pl.DataFrame:
    """Load and concatenate all available DMS datasets."""
    loaders = [load_phillips, load_engelhart, load_trastuzumab]
    dfs = []
    for loader in loaders:
        try:
            df = loader(data_dir)
            if len(df) > 0:
                # Ensure consistent schema
                if "light" not in df.columns:
                    df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("light"))
                dfs.append(df.select(["heavy", "light", "affinity", "dataset"]))
        except Exception as e:
            print(f"Warning: {loader.__name__} failed: {e}")
    return pl.concat(dfs) if dfs else pl.DataFrame()


def summarize(df: pl.DataFrame) -> None:
    print(f"Total variants: {len(df)}")
    print(df.group_by("dataset").agg(pl.len().alias("n")).sort("n", descending=True))
