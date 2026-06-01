"""Source script: ANDD 2026 — Antibody and Nanobody Design Dataset.

Processes the pre-downloaded curated_data/ANDD_dataset.xlsx.
Filters to antibodies only (Ab_or_Nano == 'Antibody') and retains
only rows with both a valid Affinity_Kd(M) and antigen sequence (Ag_Seq).

ANDD aggregates structural and affinity data from SAbDab, sdAb-DB, abYbank,
INDI, PLAbDab-nano, and curated literature SPR/BLI measurements.

Method is read from the Affinity_Method column (SPR, ITC, BLI, etc.).
Gross/fine antigen are inferred from the Ag_Name column using keyword rules.

Ref: Terrence (internal 2026 compilation). ANDD_dataset.xlsx in curated_data/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sourcing_affinities"))
from schema import (
    CENSORING_THRESHOLD_NM,
    apply_censoring,
    cast_to_schema,
    clean_seq,
    validate_df,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_KEY = "andd_2026"
CITATION = (
    "Terrence (2026) Antibody and Nanobody Design Dataset (ANDD): "
    "aggregated structural and affinity data from SAbDab, INDI, and literature."
)

XLSX_PATH = REPO_ROOT / "curated_data" / "ANDD_dataset.xlsx"
OUT_FILE = Path(__file__).parent / "andd_2026.parquet"

# Null placeholder used in the Excel file
_NULL_PLACEHOLDER = "\\"

# Lower KD floor (nM). KDs at/below this are physically implausible for a
# measured antibody-antigen interaction (≤1 pM) and are data artifacts in ANDD
# (e.g. kd_nm values down to ~1e-36). Rows with kd_nm <= this are dropped.
KD_FLOOR_NM = 0.001

# Antigen-name patterns that indicate NOT a true paratope-epitope interaction:
# antibody Fc / Fc-receptor / FcRn, bacterial Ig-binding proteins (Protein A/L),
# and Ig proteases. These bind the antibody constant region/framework, not via
# the CDR loops, so they are excluded from the affinity corpus.
_FC_RECEPTOR_EXCLUDE = re.compile(
    r"fcrn"
    r"|fc region receptor"
    r"|fc receptor"
    r"|fc-?gamma receptor"
    r"|immunoglobulin gamma fc"
    r"|\bige fc\b"
    r"|\bigg\d? ?\w* ?fc\b"
    r"|immunoglobulin heavy constant"
    r"|g[- ]binding protein a"
    r"|\bprotein l\b"
    r"|immunoglobulin a1 protease",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Gross antigen mapping from Ag_Name keywords
# ---------------------------------------------------------------------------

_GROSS_RULES: list[tuple[str, str, str]] = [
    # (pattern, gross_antigen, fine_antigen_prefix)
    (r"SARS|CoV|COVID|spike|RBD|ACE2", "SARS-CoV-2", "SARS-CoV-2"),
    (r"influenza|hemagglutinin|\bHA\b|neuraminidase|\bNA\b", "Flu", "Flu"),
    (r"HIV|gp120|gp41|gp160|Env", "HIV", "HIV"),
    (r"HER2|ErbB2|ERBB2", "HER2", "HER2-ECD"),
    (r"VEGF|vascular endothelial growth factor", "VEGF", "VEGF-A"),
    (r"TNF\b|tumor necrosis factor", "TNF", "TNF-alpha"),
    (r"IL-6|interleukin.6", "IL-6", "IL-6"),
    (r"IL-1\b|interleukin.1\b", "IL-1", "IL-1"),
    (r"EGFR|epidermal growth factor receptor", "EGFR", "EGFR-ECD"),
    (r"RSV|respiratory syncytial", "RSV", "RSV-F"),
    (r"lysozyme", "lysozyme", "lysozyme"),
    (r"insulin", "insulin", "insulin"),
    (r"PD.?1|programmed death", "PD-1", "PD-1"),
    (r"PD.?L1", "PD-L1", "PD-L1"),
    (r"CD20\b", "CD20", "CD20"),
    (r"angiopoietin|Ang2\b", "Ang2", "Ang2"),
]
_GROSS_PATTERNS = [(re.compile(p, re.IGNORECASE), g, f) for p, g, f in _GROSS_RULES]


def _infer_antigen(ag_name: Optional[str]) -> tuple[str, str]:
    """Infer (gross_antigen, fine_antigen) from the Ag_Name string."""
    if not ag_name or ag_name == _NULL_PLACEHOLDER:
        return "unknown", "unknown"
    for pat, gross, fine_prefix in _GROSS_PATTERNS:
        if pat.search(ag_name):
            return gross, ag_name.strip()[:80]
    return "other", ag_name.strip()[:80]


def _is_null(val) -> bool:
    return val is None or str(val).strip() in ("", _NULL_PLACEHOLDER, "\\\\")


def _parse_kd(val) -> Optional[float]:
    """Parse Affinity_Kd(M) to kd_nm (nM). Returns None on failure."""
    if _is_null(val):
        return None
    try:
        kd_m = float(str(val).replace(",", ".").strip())
        if kd_m <= 0:
            return None
        kd_nm = kd_m * 1e9
        if kd_nm <= KD_FLOOR_NM:  # drop sub-pM artifacts
            return None
        return kd_nm
    except (ValueError, TypeError):
        return None


def source_andd() -> pl.DataFrame:
    """Load and annotate the ANDD 2026 dataset from the Excel file."""
    print(f"\n[{SOURCE_KEY}] Loading from {XLSX_PATH}")

    import openpyxl  # type: ignore

    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    headers = [str(h) if h is not None else "" for h in next(rows)]

    required = {
        "Ab_or_Nano", "Ab/Nano H_Chain AA", "Ab/Nano L_Chain AA",
        "Ag_Seq", "Affinity_Kd(M)", "Affinity_Method", "Ag_Name",
    }
    missing = required - set(headers)
    if missing:
        raise ValueError(f"ANDD xlsx missing columns: {missing}")

    idx = {h: i for i, h in enumerate(headers)}

    records = []
    n_total = 0
    n_ab = 0
    n_kd = 0
    n_seq = 0
    n_fc = 0

    for row in rows:
        n_total += 1
        get = lambda col: row[idx[col]]  # noqa: E731

        # Filter: antibodies only
        ab_type = get("Ab_or_Nano")
        if _is_null(ab_type) or "antibody" not in str(ab_type).lower():
            continue
        n_ab += 1

        # Parse KD
        kd_nm = _parse_kd(get("Affinity_Kd(M)"))
        if kd_nm is None:
            continue
        n_kd += 1

        # Check sequences
        hc_raw = get("Ab/Nano H_Chain AA")
        lc_raw = get("Ab/Nano L_Chain AA")
        ag_raw = get("Ag_Seq")
        if _is_null(hc_raw) or _is_null(lc_raw) or _is_null(ag_raw):
            continue
        n_seq += 1

        # Parse method and antigen
        method_raw = get("Affinity_Method")
        method = str(method_raw).strip() if not _is_null(method_raw) else "unknown"

        ag_name = get("Ag_Name")
        # Drop Fc / Fc-receptor / Ig-binding-reagent "antigens" (not true Ab-Ag)
        if not _is_null(ag_name) and _FC_RECEPTOR_EXCLUDE.search(str(ag_name)):
            n_fc += 1
            continue
        gross, fine = _infer_antigen(ag_name if not _is_null(ag_name) else None)

        records.append({
            "hc_seq":          clean_seq(str(hc_raw)),
            "lc_seq":          clean_seq(str(lc_raw)),
            "target_seq":      clean_seq(str(ag_raw)),
            "kd_nm":           kd_nm,
            "kd_sd_nm":        None,
            "source_key":      SOURCE_KEY,
            "source_citation": CITATION,
            "method":          method,
            "gross_antigen":   gross,
            "fine_antigen":    fine,
            "is_censored":     False,
            "censor_type":     None,
            "notes":           f"ag_name={str(ag_name)[:60]}",
        })

    wb.close()
    print(
        f"  Rows scanned: {n_total:,} | antibody: {n_ab:,} | "
        f"has KD (>{KD_FLOOR_NM} nM): {n_kd:,} | has all seqs: {n_seq:,} | "
        f"Fc/FcR excluded: {n_fc:,}"
    )

    if not records:
        raise RuntimeError("No valid records extracted from ANDD xlsx.")

    df = pl.DataFrame(records)
    df = df.with_columns(pl.lit(False).cast(pl.Boolean).alias("is_censored"))

    # Apply censoring threshold
    df = df.with_columns([
        pl.when(pl.col("kd_nm") > CENSORING_THRESHOLD_NM)
          .then(pl.lit(True))
          .otherwise(pl.col("is_censored"))
          .alias("is_censored"),
        pl.when(pl.col("kd_nm") > CENSORING_THRESHOLD_NM)
          .then(pl.lit("upper_limit"))
          .otherwise(pl.col("censor_type"))
          .alias("censor_type"),
    ])

    df = validate_df(df, source_key=SOURCE_KEY)
    df = cast_to_schema(df)
    return df


if __name__ == "__main__":
    df = source_andd()
    print(f"\nTotal rows: {len(df):,}")
    print(df.head(3))
    print("\nGross antigen breakdown:")
    print(df.group_by("gross_antigen").len().sort("len", descending=True).head(15))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_FILE)
    print(f"\nSaved → {OUT_FILE}")
