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


def load_curated(path: Path) -> pl.DataFrame:
    """Load the curated DMS binding dataset (``filtered_data.csv`` on S3).

    File columns: ``hc_seq, lc_seq, target_seq, kd_nm, source, antibody, antigen, notes``.
    The affinity label is converted to ``-log10(KD in M)`` (higher = stronger binding),
    matching the ``neg_log_kd`` convention used elsewhere in the project.

    Args:
        path: Path to the curated CSV (e.g. ``curated_data/filtered_data.csv``).

    Returns:
        Columns: ``heavy, light, target, affinity, kd_nm, source, antibody, antigen, dataset``.
    """
    df = pl.read_csv(path, infer_schema_length=5000)
    return (
        df.rename({"hc_seq": "heavy", "lc_seq": "light", "target_seq": "target"})
        # kd_nm is in nanomolar → KD[M] = kd_nm * 1e-9 → affinity = -log10(KD[M])
        .with_columns((-(pl.col("kd_nm") * 1e-9).log10()).alias("affinity"))
        .with_columns(pl.col("source").alias("dataset"))
        .select(
            ["heavy", "light", "target", "affinity", "kd_nm",
             "source", "antibody", "antigen", "dataset"]
        )
        .drop_nulls(subset=["heavy", "light", "affinity"])
    )


def _clean_seq(s: Optional[str]) -> Optional[str]:
    """Uppercase and strip non-letter characters from a protein sequence."""
    if s is None:
        return None
    s = "".join(ch for ch in str(s).strip().upper() if ch.isalpha())
    return s if s else None


def load_andd(
    path: Path,
    sheet: str = "Sheet3",
    kd_lo: float = 1e-12,
    kd_hi: float = 1e-3,
    drop_predicted: bool = True,
    min_heavy: int = 50,
    min_antigen: int = 20,
) -> pl.DataFrame:
    """Load Terrence's unified ANDD antibody/nanobody–antigen affinity dataset.

    Keeps only rows usable for affinity regression: a heavy chain, an antigen sequence,
    and a physically sane measured KD. Predicted KDs are dropped by default.

    Args:
        path: Path to ``ANDD_dataset.xlsx``.
        sheet: Worksheet name.
        kd_lo, kd_hi: Sane KD bounds in molar (default 1 pM .. 1 mM).
        drop_predicted: Drop rows whose ``Affinity_Method`` is "Predict".
        min_heavy, min_antigen: Minimum sequence lengths to keep.

    Returns:
        Columns: ``heavy, light, target, affinity (= -log10 KD[M]), kd_nm, source,
        antibody, antigen, dataset`` — schema-compatible with :func:`load_curated`.
    """
    import numpy as np
    import pandas as pd

    pdf = pd.read_excel(path, sheet_name=sheet)
    H, L, AG, KD, M = (
        "Ab/Nano H_Chain AA", "Ab/Nano L_Chain AA", "Ag_Seq",
        "Affinity_Kd(M)", "Affinity_Method",
    )
    heavy = pdf[H].map(_clean_seq)
    light = pdf[L].map(_clean_seq)
    antigen = pdf[AG].map(_clean_seq)
    kd_m = pd.to_numeric(pdf[KD], errors="coerce")
    method = pdf[M].astype(str).str.strip()

    mask = (
        heavy.fillna("").str.len().ge(min_heavy)
        & antigen.fillna("").str.len().ge(min_antigen)
        & kd_m.between(kd_lo, kd_hi)
    )
    if drop_predicted:
        mask &= ~method.str.lower().str.startswith("predict")
    mask = mask.fillna(False)

    out = pd.DataFrame({
        "heavy": heavy[mask],
        "light": light[mask],
        "target": antigen[mask],
        "affinity": -np.log10(kd_m[mask].astype(float)),          # pKd
        "kd_nm": kd_m[mask].astype(float) * 1e9,
        "source": "ANDD:" + pdf.loc[mask, "Source"].astype(str),
        "antibody": pdf.loc[mask, "Ab_or_Nano"].astype(str),
        "antigen": pdf.loc[mask, "Ag_Name"].astype(str).replace("", "unknown"),
        "dataset": "ANDD",
    })
    return pl.from_pandas(out)


def normalize_affinity(
    df: pl.DataFrame,
    by: tuple[str, ...] = ("source", "antigen"),
    method: str = "zscore",
    out_col: str = "affinity_norm",
) -> pl.DataFrame:
    """Normalize the affinity label *within* experimental groups.

    The curated dataset pools assays with different scales, dynamic-range ceilings, and
    antigens, so raw ``-log10(KD)`` is not comparable across rows. This rescales the
    ``affinity`` column within each ``by`` group so a single regression sees a consistent
    target. Use this when pooling sources/antigens; skip it if you train on one group.

    Singleton / zero-variance groups are left centered at 0 (z-score) to avoid NaNs.

    Args:
        df: Output of :func:`load_curated` (must contain an ``affinity`` column).
        by: Grouping columns defining a comparable assay/antigen unit.
        method: ``"zscore"`` (per-group mean 0 / std 1) or ``"rank"`` (per-group quantile in [0, 1]).
        out_col: Name of the added normalized column.

    Returns:
        ``df`` with an added ``out_col`` column (raw ``affinity`` is preserved).
    """
    by = list(by)
    if method == "zscore":
        std = pl.col("affinity").std().over(by)
        return df.with_columns(
            pl.when(std > 0)
            .then((pl.col("affinity") - pl.col("affinity").mean().over(by)) / std)
            .otherwise(0.0)
            .alias(out_col)
        )
    if method == "rank":
        n = pl.len().over(by)
        return df.with_columns(
            pl.when(n > 1)
            .then((pl.col("affinity").rank("average").over(by) - 1) / (n - 1))
            .otherwise(0.5)
            .alias(out_col)
        )
    raise ValueError(f"Unknown method '{method}'. Use 'zscore' or 'rank'.")


def load_unified(path: Path) -> pl.DataFrame:
    """Load the prebuilt unified affinity dataset (output of build_unified_affinity.py).

    Columns: ``heavy, light, target, affinity, affinity_norm, source, antigen, norm_group, dataset``.
    """
    return pl.read_csv(path, infer_schema_length=5000)


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
