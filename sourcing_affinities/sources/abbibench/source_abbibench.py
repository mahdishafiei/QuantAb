"""Source script: AbBiBench (Antibody Binding Benchmark) HuggingFace release.

AbBiBench aggregates many antibody-antigen binding datasets. We ingest only the
systems that are NOT already covered natively by our other sources:

  Already in corpus (SKIPPED here to avoid duplication):
    - 3gbn / 4fqi  → phillips_2021   (CR6261 / CR9114 anti-influenza)
    - aayl49/50/51/52, aayl49_ML → engelhart_2022 (AlphaSeq anti-SARS-CoV-2 Spike)

  New systems ingested below (0% sequence overlap with the existing corpus):
    | AbBiBench | antibody / antigen        | binding_score | measure_type   |
    |-----------|---------------------------|---------------|----------------|
    | 1mhp      | AQC2 / integrin-α1        | -log10(KD)    | kd             |
    | 1n8z      | trastuzumab / HER2        | -log10(KD)    | kd             |
    | 2fjg,g6   | G6.31 / VEGF-A            | log enrich.   | log_enrichment |
    | 1mlc      | D44.1 / lysozyme          | log enrich.   | log_enrichment |
    | 5a12_ang2 | 5A12 / Angiopoietin-2     | log enrich.   | log_enrichment |
    | 5a12_vegf | 5A12 / VEGF-A             | log enrich.   | log_enrichment |
    | 4d5_her2  | trastuzumab(4D5) / HER2   | log enrich.   | log_enrichment |

Per the AbBiBench paper, binding_score is standardized as -log10(KD) for KD
assays or log(enrichment ratio) for display selections — higher = stronger in
both cases. KD-assay rows get kd_nm = 10^(9 - binding_score); enrichment rows
keep kd_nm null and carry the score only in binding_score (+ measure_type).

Antibody heavy/light sequences come straight from each benchmarking CSV. Antigen
(target_seq) is read from the antigen chain of the complex PDB (ATOM records).

Data: HuggingFace AbBibench/Antibody_Binding_Benchmark_Dataset, mirrored under
      raw/. Primary studies cited per system below.
Paper: AbBiBench: A Benchmark for Antibody Binding Affinity Maturation and
       Design (2025). arXiv:2506.04235.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sourcing_affinities"))
from schema import (  # noqa: E402
    apply_censoring,
    cast_to_schema,
    extract_pdb_chain,
    validate_df,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SOURCE_DIR = Path(__file__).parent
RAW_DIR = SOURCE_DIR / "raw"
CSV_DIR = RAW_DIR / "binding_affinity"
PDB_DIR = RAW_DIR / "complex_structure"
OUT_FILE = SOURCE_DIR / "abbibench.parquet"

ABBIBENCH_PROV = "via AbBiBench (arXiv:2506.04235)"

# Per-system citations (primary study where known, else AbBiBench).
CIT = {
    "clark_2006": (
        "Clark LA et al. (2006) Affinity enhancement of an in vivo matured "
        "therapeutic antibody using structure-based computational design. "
        f"Protein Sci 15(5):949-960. doi:10.1110/ps.052030506 [{ABBIBENCH_PROV}]"
    ),
    "koenig_2017": (
        "Koenig P et al. (2017) Mutational landscape of antibody variable domains "
        "reveals a switch between stability and binding optimization at cold "
        f"temperatures. PNAS 114(3):497-502. doi:10.1073/pnas.1613231114 "
        f"[{ABBIBENCH_PROV}]"
    ),
    "warszawski_2020": (
        "Warszawski S et al. (2020) Optimizing antibody affinity and stability by "
        "the automated design of the variable light-heavy chain interfaces. "
        f"PLOS Comput Biol 16(4):e1007828. doi:10.1371/journal.pcbi.1007828 "
        f"[{ABBIBENCH_PROV}]"
    ),
    "minot_2024": (
        "Minot M & Reddy ST (2024) Deep mutational scanning of antibody 5A12 "
        f"against Angiopoietin-2 / VEGF. [{ABBIBENCH_PROV}]"
    ),
    "abbibench_her2": (
        "Trastuzumab / HER2 binding dataset; primary study per AbBiBench Table 1. "
        "AbBiBench: A Benchmark for Antibody Binding Affinity Maturation and "
        "Design (2025). arXiv:2506.04235."
    ),
}


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AbDataset:
    """One AbBiBench benchmarking file mapped onto the corpus schema."""

    csv: str            # filename under raw/binding_affinity/
    source_key: str
    gross_antigen: str
    fine_antigen: str
    method: str
    measure_type: str   # "kd" | "log_enrichment"
    pdb: str            # filename stem under raw/complex_structure/ (no .pdb)
    ag_chain: str
    citation_key: str


DATASETS: list[AbDataset] = [
    # --- calibrated KD systems ---
    AbDataset("1mhp_LC_benchmarking_data.csv", "clark_2006",
              "Integrin", "Integrin-alpha1-I-domain", "phage_display",
              "kd", "1mhp_hla", "A", "clark_2006"),
    AbDataset("1n8z_benchmarking_data.csv", "abbibench_1n8z_her2",
              "HER2", "HER2-ECD", "display",
              "kd", "1n8z_bac", "C", "abbibench_her2"),
    # --- log-enrichment systems ---
    AbDataset("2fjg_benchmarking_data.csv", "koenig_2017",
              "VEGF", "VEGF-A", "phage_display",
              "log_enrichment", "2fjg_hlv", "V", "koenig_2017"),
    AbDataset("g6_LC_benchmarking_data.csv", "koenig_2017",
              "VEGF", "VEGF-A", "phage_display",
              "log_enrichment", "2fjg_hlv", "V", "koenig_2017"),
    AbDataset("1mlc_benchmarking_data.csv", "warszawski_2020",
              "Lysozyme", "hen-egg-lysozyme", "yeast_display",
              "log_enrichment", "1mlc_bae", "E", "warszawski_2020"),
    AbDataset("5a12_ang2_benchmarking_data.csv", "minot_2024",
              "Ang2", "Angiopoietin-2-FLD", "yeast_display",
              "log_enrichment", "4zfg_hla", "A", "minot_2024"),
    AbDataset("5a12_vegf_benchmarking_data.csv", "minot_2024",
              "VEGF", "VEGF-A", "yeast_display",
              "log_enrichment", "4zff_hld", "D", "minot_2024"),
    AbDataset("4d5_her2_benchmarking_data_trimmed.csv", "abbibench_4d5_her2",
              "HER2", "HER2-ECD", "display",
              "log_enrichment", "1n8z_bac", "C", "abbibench_her2"),
]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def _ingest_one(ds: AbDataset) -> pl.DataFrame:
    """Map a single AbBiBench benchmarking CSV onto the corpus schema."""
    df = pl.read_csv(CSV_DIR / ds.csv)
    target_seq = extract_pdb_chain(PDB_DIR / f"{ds.pdb}.pdb", ds.ag_chain)
    if len(target_seq) < 30:
        raise ValueError(
            f"{ds.csv}: antigen chain {ds.ag_chain} of {ds.pdb} gave only "
            f"{len(target_seq)} residues — check PDB / chain id."
        )

    is_kd = ds.measure_type == "kd"
    # binding_score is -log10(KD/M) (pKD) for KD assays → kd_nm = 10^(9 - pKD)
    kd_expr = (
        (10.0 ** (9.0 - pl.col("binding_score")))
        if is_kd
        else pl.lit(None, dtype=pl.Float64)
    )

    df = df.with_columns([
        pl.col("heavy_chain_seq").alias("hc_seq"),
        pl.col("light_chain_seq").alias("lc_seq"),
        pl.lit(target_seq).alias("target_seq"),
        kd_expr.alias("kd_nm"),
        pl.lit(None, dtype=pl.Float64).alias("kd_sd_nm"),
        pl.col("binding_score").cast(pl.Float64).alias("binding_score"),
        pl.lit(ds.measure_type).alias("measure_type"),
        pl.lit(ds.source_key).alias("source_key"),
        pl.lit(CIT[ds.citation_key]).alias("source_citation"),
        pl.lit(ds.method).alias("method"),
        pl.lit(ds.gross_antigen).alias("gross_antigen"),
        pl.lit(ds.fine_antigen).alias("fine_antigen"),
        pl.lit(False).alias("is_censored"),
        pl.lit(None, dtype=pl.Utf8).alias("censor_type"),
        pl.lit(f"AbBiBench {ds.csv.replace('_benchmarking_data', '').replace('.csv', '')}; "
               f"antigen={ds.pdb}:{ds.ag_chain}").alias("notes"),
    ])

    df = apply_censoring(df)   # only bites KD rows; null kd_nm stays uncensored
    df = validate_df(df, source_key=ds.source_key)
    df = cast_to_schema(df)
    return df


def source_abbibench() -> pl.DataFrame:
    """Ingest all configured new AbBiBench systems into one schema-cast frame."""
    print(f"\n[abbibench] ingesting {len(DATASETS)} benchmarking files from {CSV_DIR}")
    parts = [_ingest_one(ds) for ds in tqdm(DATASETS, desc="abbibench")]
    df = pl.concat(parts)
    print("\nPer source_key / measure_type:")
    print(
        df.group_by(["source_key", "measure_type", "gross_antigen"])
        .len()
        .sort("len", descending=True)
    )
    return df


if __name__ == "__main__":
    df = source_abbibench()
    print(f"\nTotal rows: {len(df):,}")
    print(df.select(["source_key", "gross_antigen", "measure_type",
                     "kd_nm", "binding_score", "is_censored"]).head(5))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_FILE)
    print(f"\nSaved → {OUT_FILE}")
