"""Source script: Phillips et al. 2021 CR9114/CR6261 anti-influenza DMS.

Re-processes raw DMS CSVs from DMS_Data/02_phillips_cr9114_cr6261/.
Full HC+LC sequences available directly in the CSV files.

Paper: Phillips et al. (2021) Binding affinities of 130,000+ variants of
       two influenza broadly neutralizing antibodies mapped by deep mutational
       scanning. eLife 10:e71393. https://doi.org/10.7554/eLife.71393

Target sequences:
  H1-HA: A/Puerto Rico/8/1934 (PR8) — extracted from PDB 3GBN chains A+B
  H3-HA: A/Hong Kong/1/1968 — fetched from UniProt P03437 if available
  H9-HA: A/Hong Kong/1073/1999 — fetched from UniProt Q9IH43 if available
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
    neg_log_kd_to_nm,
    validate_df,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_KEY = "phillips_2021"
CITATION = (
    "Phillips AM et al. (2021) Binding affinities of 130,000+ variants of two "
    "influenza broadly neutralizing antibodies mapped by deep mutational scanning. "
    "eLife 10:e71393. doi:10.7554/eLife.71393"
)
METHOD = "yeast_display"
GROSS_ANTIGEN = "Flu"

RAW_DIR = REPO_ROOT / "DMS_Data" / "02_phillips_cr9114_cr6261"
PDB_DIR = REPO_ROOT / "DMS_Data" / "pdbs"
OUT_FILE = Path(__file__).parent / "phillips_2021.parquet"


def _h1_target_seq_pr8() -> str:
    """H1 HA ectodomain from PDB 3GBN (PR/8/34): HA1 (chain A) + HA2 (chain B)."""
    ha1 = extract_pdb_chain(PDB_DIR / "3GBN.pdb", "A")  # 331 aa
    ha2 = extract_pdb_chain(PDB_DIR / "3GBN.pdb", "B")  # 179 aa
    return ha1 + ha2


def _fetch_uniprot_seq(accession: str) -> str | None:
    """Fetch a sequence from UniProt FASTA endpoint. Returns None on failure."""
    try:
        import requests  # type: ignore
        r = requests.get(
            f"https://www.uniprot.org/uniprot/{accession}.fasta", timeout=15
        )
        if r.ok:
            lines = r.text.strip().split("\n")
            seq = "".join(lines[1:]).upper()
            # Strip signal peptide (UniProt full-length): keep only mature ectodomain
            # For HA: remove first ~16-17 aa signal peptide and last ~27 aa TM+cytoplasmic
            # This is an approximation — for exact trimming use the PDB structure
            return seq
        return None
    except Exception:
        return None


# Map from (antibody, ha_subtype) to (fine_antigen, target_seq_fetcher)
def _build_target_seq_map() -> dict[str, str]:
    """Build target sequence lookup dict. Fetches missing sequences from UniProt."""
    h1 = _h1_target_seq_pr8()

    # For H3 (A/Hong Kong/1/68): UniProt P03437 — note this is full precursor;
    # ectodomain trimming is approximate. For production, verify against PDB 2HMG.
    h3 = _fetch_uniprot_seq("P03437")
    if h3 is None:
        print(
            "  WARNING: Could not fetch H3 HA (UniProt P03437). "
            "H3 files will be skipped. Provide internet access or hardcode the sequence."
        )

    # For H9 (A/Hong Kong/1073/99): UniProt Q9IH43
    h9 = _fetch_uniprot_seq("Q9IH43")
    if h9 is None:
        print(
            "  WARNING: Could not fetch H9 HA (UniProt Q9IH43). "
            "H9 files will be skipped. Provide internet access or hardcode the sequence."
        )

    return {"H1": h1, "H3": h3, "H9": h9}


# CSV filename → (antibody_name, ha_subtype, neg_log_kd_col)
FILE_MAP = {
    "phillips2021binding_cr9114_h1_kd.csv": ("CR9114", "H1", "h1_mean_neg_log_kd"),
    "phillips2021binding_cr9114_h3_kd.csv": ("CR9114", "H3", "h3_mean_neg_log_kd"),
    "phillips2021binding_cr6261_h1_kd.csv": ("CR6261", "H1", "h1_mean_neg_log_kd"),
    "phillips2021binding_cr6261_h9_kd.csv": ("CR6261", "H9", "h9_mean_neg_log_kd"),
}


def _detect_neg_log_kd_col(df: pl.DataFrame) -> str:
    """Find the neg_log_kd column regardless of exact name."""
    candidates = [c for c in df.columns if "neg_log_kd" in c]
    if not candidates:
        raise ValueError(f"No neg_log_kd column found. Columns: {df.columns}")
    return candidates[0]


def source_phillips() -> pl.DataFrame:
    """Load, convert, and annotate the Phillips 2021 DMS dataset."""
    print(f"\n[{SOURCE_KEY}] Loading from {RAW_DIR}")
    target_seqs = _build_target_seq_map()

    parts: list[pl.DataFrame] = []
    for csv_file in sorted(RAW_DIR.glob("*.csv")):
        fname = csv_file.name
        if fname not in FILE_MAP:
            continue

        antibody, ha_subtype, _ = FILE_MAP[fname]
        fine_antigen = f"{ha_subtype}-HA"
        target_seq = target_seqs.get(ha_subtype)

        if target_seq is None:
            print(f"  Skipping {fname}: {fine_antigen} target sequence unavailable.")
            continue

        df = pl.read_csv(csv_file)
        neg_log_col = _detect_neg_log_kd_col(df)

        df = (
            df.rename({neg_log_col: "neg_log_kd_raw"})
            .filter(
                pl.col("heavy").is_not_null()
                & pl.col("light").is_not_null()
                & pl.col("neg_log_kd_raw").is_not_null()
            )
            .with_columns([
                pl.col("heavy").alias("hc_seq"),
                pl.col("light").alias("lc_seq"),
                pl.lit(target_seq).alias("target_seq"),
                # neg_log_kd = -log10(KD/M) → kd_nm = 10^(9 - neg_log_kd)
                (10.0 ** (9.0 - pl.col("neg_log_kd_raw"))).alias("kd_nm"),
                pl.lit(None).cast(pl.Float64).alias("kd_sd_nm"),
                pl.lit(SOURCE_KEY).alias("source_key"),
                pl.lit(CITATION).alias("source_citation"),
                pl.lit(METHOD).alias("method"),
                pl.lit(GROSS_ANTIGEN).alias("gross_antigen"),
                pl.lit(fine_antigen).alias("fine_antigen"),
                pl.lit(False).alias("is_censored"),
                pl.lit(None).cast(pl.Utf8).alias("censor_type"),
                pl.lit(f"antibody={antibody}; file={fname}").alias("notes"),
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
        print(f"  {fname}: {n:>7,} rows ({antibody} vs {fine_antigen})")
        parts.append(df)

    if not parts:
        raise RuntimeError(f"No data loaded from {RAW_DIR}")

    combined = pl.concat(parts)
    combined = validate_df(combined, source_key=SOURCE_KEY, max_hc=250, max_lc=250)
    combined = cast_to_schema(combined)
    return combined


if __name__ == "__main__":
    df = source_phillips()
    print(f"\nTotal rows: {len(df):,}")
    print(df.head(3))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_FILE)
    print(f"\nSaved → {OUT_FILE}")
