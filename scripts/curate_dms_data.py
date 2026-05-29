#!/usr/bin/env python3
"""
Curate raw DMS datasets into a standardised schema.

For each dataset, writes curated_data/<name>_curated.csv with columns:
    hc_seq, lc_seq, target_seq, kd_nm, source, antibody, antigen, notes

Then merges all into curated_data/all_data.csv.

Run from the repo root:
    python scripts/curate_dms_data.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import polars as pl
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "DMS_Data"
PDBS_DIR = DATA_DIR / "pdbs"
OUT_DIR = REPO / "curated_data"
OUT_DIR.mkdir(exist_ok=True)

SCHEMA = {
    "hc_seq": pl.Utf8,
    "lc_seq": pl.Utf8,
    "target_seq": pl.Utf8,
    "kd_nm": pl.Float64,
    "source": pl.Utf8,
    "antibody": pl.Utf8,
    "antigen": pl.Utf8,
    "notes": pl.Utf8,
}

# ---------------------------------------------------------------------------
# PDB utilities
# ---------------------------------------------------------------------------
AA3TO1: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U",
}


def extract_seqres(pdb_path: Path, chain: str) -> str:
    """Extract sequence from SEQRES records for the given chain.

    Args:
        pdb_path: Path to PDB file.
        chain: Single-letter chain identifier.

    Returns:
        One-letter amino acid sequence; unknown residues mapped to 'X'.
    """
    residues: list[str] = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("SEQRES") and line[11] == chain:
                residues.extend(line[19:].split())
    return "".join(AA3TO1.get(r, "X") for r in residues)


def extract_atom_kabat(
    pdb_path: Path,
    chain: str,
    max_kabat: int = 999,
) -> dict[tuple[int, str], str]:
    """Extract residues from ATOM records keyed by (resnum, insertion_code).

    Antibody PDB files use Kabat numbering in the residue sequence number
    column, so the returned keys are Kabat positions.

    Args:
        pdb_path: Path to PDB file.
        chain: Single-letter chain identifier.
        max_kabat: Discard residues with resnum > max_kabat (trims constant domains).

    Returns:
        Dict mapping (Kabat_resnum, insertion_code) → one-letter AA.
        Insertion codes are empty string for most positions, 'A'/'B'/... for
        CDR-loop insertions (e.g. Kabat 100A, 100B).
    """
    residues: dict[tuple[int, str], str] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            resname = line[17:20].strip()
            aa = AA3TO1.get(resname)
            if aa is None:
                continue
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            if resnum > max_kabat:
                continue
            icode = line[26].strip()
            key = (resnum, icode)
            if key not in residues:
                residues[key] = aa
    return residues


def kabat_dict_to_seq(residues: dict[tuple[int, str], str]) -> str:
    """Reconstruct sequence by sorting (resnum, icode) Kabat keys.

    Args:
        residues: Dict mapping (resnum, icode) → one-letter AA.

    Returns:
        Amino acid sequence sorted by Kabat position.
    """
    sorted_keys = sorted(residues.keys(), key=lambda k: (k[0], k[1]))
    return "".join(residues[k] for k in sorted_keys)


def apply_single_kabat_mut(
    wt_dict: dict[tuple[int, str], str],
    mut_str: str,
) -> dict[tuple[int, str], str]:
    """Apply a single mutation string like 'M59I' or 'A100AI' (Kabat pos 100A).

    Format: <WT_aa><resnum>[icode]<mut_aa>   e.g. 'A33A', 'M59I', 'G100AI'

    Args:
        wt_dict: Kabat-keyed WT residue dictionary.
        mut_str: Single mutation string.

    Returns:
        New dict with the mutation applied (original is not modified).
    """
    result = dict(wt_dict)
    m = re.match(r"^([A-Z*])(\d+)([A-Z]?)([A-Z*])$", mut_str.strip())
    if m:
        _wt_aa, resnum_str, icode, mut_aa = m.groups()
        key = (int(resnum_str), icode)
        if key in result:
            result[key] = mut_aa
    return result


def apply_kabat_mut_list(
    wt_dict: dict[tuple[int, str], str],
    muts_str: str,
) -> dict[tuple[int, str], str]:
    """Apply a semicolon-separated list of Kabat mutations.

    Args:
        wt_dict: Kabat-keyed WT residue dictionary.
        muts_str: Semicolon-separated mutations, e.g. 'M59I;T98T;Q120K'.

    Returns:
        New dict with all mutations applied.
    """
    result = dict(wt_dict)
    for mut in muts_str.split(";"):
        mut = mut.strip()
        if mut:
            result = apply_single_kabat_mut(result, mut)
    return result


# ---------------------------------------------------------------------------
# Target sequences from PDB (built once, reused)
# ---------------------------------------------------------------------------
def _load_ha_targets() -> dict[str, str]:
    """Load H1 and H3 influenza HA target sequences from PDB SEQRES.

    Returns:
        Dict with keys 'H1', 'H3' mapping to concatenated HA1+HA2 sequences.
    """
    targets: dict[str, str] = {}
    h1_pdb = PDBS_DIR / "4FQI.pdb"
    h3_pdb = PDBS_DIR / "3GBN.pdb"
    if h1_pdb.exists():
        ha1 = extract_seqres(h1_pdb, "A")
        ha2 = extract_seqres(h1_pdb, "B")
        targets["H1"] = ha1 + ha2
    if h3_pdb.exists():
        ha1 = extract_seqres(h3_pdb, "A")
        ha2 = extract_seqres(h3_pdb, "B")
        targets["H3"] = ha1 + ha2
    return targets


def _load_sars_rbd() -> str:
    """Load SARS-CoV-2 RBD sequence from 6XC3 chain C."""
    pdb = PDBS_DIR / "6XC3.pdb"
    return extract_seqres(pdb, "C") if pdb.exists() else ""


def _load_her2() -> str:
    """Load HER2 sequence from 1N8Z chain C."""
    pdb = PDBS_DIR / "1N8Z.pdb"
    return extract_seqres(pdb, "C") if pdb.exists() else ""


# ---------------------------------------------------------------------------
# Dataset 2 — Phillips et al. 2021
# ---------------------------------------------------------------------------
def curate_phillips() -> pl.DataFrame:
    """Curate Phillips et al. 2021 CR9114/CR6261 DMS data.

    Four files covering two antibodies (CR9114, CR6261) against two antigens
    each (H1/H3 for CR9114; H1/H9 for CR6261). Full VH+VL sequences provided.

    Returns:
        Standardised DataFrame.
    """
    src = DATA_DIR / "02_phillips_cr9114_cr6261"
    ha = _load_ha_targets()

    # (filename, antibody, antigen, kd_col, target_key)
    files: list[tuple[str, str, str, str, Optional[str]]] = [
        ("phillips2021binding_cr9114_h1_kd.csv", "CR9114", "H1-HA", "h1_mean_neg_log_kd", "H1"),
        ("phillips2021binding_cr9114_h3_kd.csv", "CR9114", "H3-HA", "h3_mean", "H3"),
        ("phillips2021binding_cr6261_h1_kd.csv", "CR6261", "H1-HA", "h1_mean_neg_log_kd", "H1"),
        ("phillips2021binding_cr6261_h9_kd.csv", "CR6261", "H9-HA", "h9_mean_neg_log_kd", None),
    ]

    frames: list[pl.DataFrame] = []
    for fname, ab, antigen, kd_col, tgt_key in tqdm(files, desc="Phillips"):
        fpath = src / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping", file=sys.stderr)
            continue
        df = pl.read_csv(fpath)
        target_seq = ha.get(tgt_key, None) if tgt_key else None
        notes = "" if target_seq else f"no PDB for {antigen} target"
        frames.append(
            pl.DataFrame({
                "hc_seq": df["heavy"],
                "lc_seq": df["light"],
                "target_seq": [target_seq] * len(df),
                "kd_nm": (10.0 ** (9.0 - df[kd_col])).cast(pl.Float64),
                "source": ["phillips2021"] * len(df),
                "antibody": [ab] * len(df),
                "antigen": [antigen] * len(df),
                "notes": [notes] * len(df),
            })
        )

    return pl.concat(frames)


# ---------------------------------------------------------------------------
# Dataset 1 — Dailey et al. (CR9114 single-mutant DMS)
# ---------------------------------------------------------------------------
def curate_dailey() -> pl.DataFrame:
    """Curate Dailey CR9114 single-point mutant DMS data.

    Mutations given in Kabat notation relative to WT CR9114. WT VH and VL
    sequences reconstructed from 4FQI (H1 complex) ATOM records.
    Only H1 and H3 targets included (PDB available); GL and fluB omitted.

    Returns:
        Standardised DataFrame.
    """
    src = DATA_DIR / "01_dailey_cr9114"
    ha = _load_ha_targets()

    h1_pdb = src / "4FQI.pdb"
    h3_pdb = PDBS_DIR / "3GBN.pdb"

    if not h1_pdb.exists():
        print("  WARNING: Dailey 4FQI.pdb not found, skipping", file=sys.stderr)
        return pl.DataFrame(schema=SCHEMA)

    # VH Kabat 1-113, VL Kabat 1-107 (standard domain boundaries)
    wt_vh = extract_atom_kabat(h1_pdb, "H", max_kabat=113)
    wt_vl = extract_atom_kabat(h1_pdb, "L", max_kabat=107)

    if not wt_vh:
        print("  WARNING: No ATOM records for chain H in 4FQI.pdb", file=sys.stderr)
        return pl.DataFrame(schema=SCHEMA)

    # (filename, antigen, target_key)
    files: list[tuple[str, str, str]] = [
        ("HC_WT_H1_KD_table_summary_kabat.csv", "H1-HA", "H1"),
        ("HC_WT_H3_KD_table_summary_kabat.csv", "H3-HA", "H3"),
    ]

    rows: list[dict] = []
    for fname, antigen, tgt_key in tqdm(files, desc="Dailey"):
        fpath = src / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping", file=sys.stderr)
            continue
        df = pl.read_csv(fpath)
        target_seq = ha.get(tgt_key)

        for row in df.iter_rows(named=True):
            mut_kabat = row.get("mut_kabat", "") or ""
            kd_val = row.get("minus_log_Kd")
            if kd_val is None:
                continue
            try:
                kd_val = float(kd_val)
            except (TypeError, ValueError):
                continue

            mut_vh = apply_single_kabat_mut(wt_vh, mut_kabat)
            hc_seq = kabat_dict_to_seq(mut_vh)
            lc_seq = kabat_dict_to_seq(wt_vl)
            rows.append({
                "hc_seq": hc_seq,
                "lc_seq": lc_seq,
                "target_seq": target_seq,
                "kd_nm": 10.0 ** (9.0 - kd_val),
                "source": "dailey",
                "antibody": "CR9114",
                "antigen": antigen,
                "notes": f"mut={mut_kabat}",
            })

    return pl.DataFrame(rows, schema=SCHEMA) if rows else pl.DataFrame(schema=SCHEMA)


# ---------------------------------------------------------------------------
# Dataset 3 — MAGMA-seq (4A8 and CC12.1 / CC121 SARS-CoV-2 DMS)
# ---------------------------------------------------------------------------
def _parse_magma_variant(
    variant_str: str,
    vh_wt: dict[tuple[int, str], str],
    vl_wt: dict[tuple[int, str], str],
) -> tuple[str, str]:
    """Parse MAGMA-seq variant string and return mutated (hc_seq, lc_seq).

    Variant format:
        ``<Ab>>VH:<mut1>-<codon>;<mut2>-<codon>|<Ab>>VL:<mut3>-<codon>``

    The codon suffix after '-' is discarded; only AA mutations are used.

    Args:
        variant_str: Raw variant string from SD5 file.
        vh_wt: Kabat-keyed WT VH residue dict.
        vl_wt: Kabat-keyed WT VL residue dict.

    Returns:
        Tuple of (hc_seq, lc_seq) after applying mutations.
    """
    vh_dict = dict(vh_wt)
    vl_dict = dict(vl_wt)

    for part in variant_str.split("|"):
        if ">VH:" in part:
            muts_raw = part.split(">VH:", 1)[1]
        elif ">VL:" in part:
            muts_raw = part.split(">VL:", 1)[1]
        else:
            continue

        # Strip codon suffixes: 'M59I-ATC' → 'M59I'
        muts = ";".join(m.split("-")[0] for m in muts_raw.split(";"))
        if ">VH:" in part:
            vh_dict = apply_kabat_mut_list(vh_dict, muts)
        else:
            vl_dict = apply_kabat_mut_list(vl_dict, muts)

    return kabat_dict_to_seq(vh_dict), kabat_dict_to_seq(vl_dict)


def curate_magma_seq() -> pl.DataFrame:
    """Curate MAGMA-seq 4A8 / CC12.1 SARS-CoV-2 DMS data.

    WT sequences:
      4A8   VH/VL → 7C2L chains H / L   (ATOM records, Kabat positions ≤ 113/107)
      CC121 VH/VL → 6XC3 chains B / A   (ATOM records, Kabat positions ≤ 113/107)
    Target: SARS-CoV-2 RBD → 6XC3 chain C (SEQRES)

    Note: 4A8 binds the spike NTD; the RBD (6XC3 chain C) is used as the
    closest available target sequence in our PDB collection.

    Returns:
        Standardised DataFrame.
    """
    src = DATA_DIR / "03_magma_seq"
    fpath = src / "SD5_4A8_CC121_combined.csv"
    if not fpath.exists():
        print("  WARNING: MAGMA-seq combined CSV not found", file=sys.stderr)
        return pl.DataFrame(schema=SCHEMA)

    pdb_4a8 = PDBS_DIR / "7C2L.pdb"
    pdb_cc121 = PDBS_DIR / "6XC3.pdb"

    wt_seqs: dict[str, tuple[dict, dict]] = {}

    if pdb_4a8.exists():
        vh = extract_atom_kabat(pdb_4a8, "H", max_kabat=113)
        vl = extract_atom_kabat(pdb_4a8, "L", max_kabat=107)
        wt_seqs["4A8"] = (vh, vl)
    else:
        print("  WARNING: 7C2L.pdb not found; 4A8 rows will be skipped", file=sys.stderr)

    if pdb_cc121.exists():
        # 6XC3: chain B = CC12.1 VH, chain A = CC12.1 VL
        vh = extract_atom_kabat(pdb_cc121, "B", max_kabat=113)
        vl = extract_atom_kabat(pdb_cc121, "A", max_kabat=107)
        wt_seqs["CC121"] = (vh, vl)
    else:
        print("  WARNING: 6XC3.pdb not found; CC121 rows will be skipped", file=sys.stderr)

    target_seq = _load_sars_rbd()
    df = pl.read_csv(fpath)

    rows: list[dict] = []
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc="MAGMA-seq"):
        ab = row.get("Ab", "")
        if ab not in wt_seqs:
            continue
        variant = row.get("Variant", "")
        kd_val = row.get("Kd")
        if kd_val is None:
            continue
        try:
            kd_nm = float(kd_val)
        except (TypeError, ValueError):
            continue

        vh_wt, vl_wt = wt_seqs[ab]
        try:
            hc_seq, lc_seq = _parse_magma_variant(variant, vh_wt, vl_wt)
        except Exception as exc:
            hc_seq, lc_seq = "", ""
            ab_note = f"variant_parse_error: {exc}"
        else:
            ab_note = ""

        notes = "4A8 binds NTD; RBD used as proxy target" if ab == "4A8" else ""
        if ab_note:
            notes = ab_note

        rows.append({
            "hc_seq": hc_seq,
            "lc_seq": lc_seq,
            "target_seq": target_seq,
            "kd_nm": kd_nm,
            "source": "magma_seq",
            "antibody": ab,
            "antigen": "SARS-CoV-2",
            "notes": notes,
        })

    return pl.DataFrame(rows, schema=SCHEMA) if rows else pl.DataFrame(schema=SCHEMA)


# ---------------------------------------------------------------------------
# Dataset 5 — Engelhart et al. (AAYL self-binding synthetic antibodies)
# ---------------------------------------------------------------------------
def curate_engelhart() -> pl.DataFrame:
    """Curate Engelhart AAYL synthetic antibody DMS data.

    No protein target (self-binding assay). Binding score treated as
    -log10(KD/M). All four AAYL benchmarking CSVs are combined.

    Returns:
        Standardised DataFrame.
    """
    src = DATA_DIR / "05_engelhart_aayl"
    csv_files = [
        "aayl49_benchmarking_data.csv",
        "aayl50_LC_benchmarking_data.csv",
        "aayl51_benchmarking_data.csv",
        "aayl52_LC_benchmarking_data.csv",
    ]

    frames: list[pl.DataFrame] = []
    for fname in tqdm(csv_files, desc="Engelhart"):
        fpath = src / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping", file=sys.stderr)
            continue
        df = pl.read_csv(fpath)
        frames.append(
            pl.DataFrame({
                "hc_seq": df["heavy_chain_seq"],
                "lc_seq": df["light_chain_seq"],
                "target_seq": pl.Series([None] * len(df), dtype=pl.Utf8),
                "kd_nm": (10.0 ** (9.0 - df["binding_score"])).cast(pl.Float64),
                "source": pl.Series([fname.replace(".csv", "")] * len(df)),
                "antibody": pl.Series(["AAYL"] * len(df)),
                "antigen": pl.Series(["self-binding"] * len(df)),
                "notes": pl.Series(["binding_score=-log10(KD_M); no protein target"] * len(df)),
            })
        )

    return pl.concat(frames) if frames else pl.DataFrame(schema=SCHEMA)


# ---------------------------------------------------------------------------
# Dataset 6 — Shanehsazzadeh et al. (HER2 HCDR3-only)
# ---------------------------------------------------------------------------
def curate_shanehsazzadeh() -> pl.DataFrame:
    """Curate Shanehsazzadeh HER2 binder data (HCDR3 only).

    Only HCDR3 sequences are available; full VH/VL are absent.
    hc_seq is set to the HCDR3 string with a note; lc_seq is null.
    Target: HER2 extracellular domain from 1N8Z chain C.

    Returns:
        Standardised DataFrame with hc_seq=HCDR3, lc_seq=null.
    """
    fpath = (
        DATA_DIR
        / "06_shanehsazzadeh_trastuzumab"
        / "unlocking-de-novo-antibody-design"
        / "zero-shot-binders.csv"
    )
    if not fpath.exists():
        print("  WARNING: Shanehsazzadeh zero-shot-binders.csv not found", file=sys.stderr)
        return pl.DataFrame(schema=SCHEMA)

    df = pl.read_csv(fpath)
    target_seq = _load_her2()

    return pl.DataFrame({
        "hc_seq": df["HCDR3"],
        "lc_seq": pl.Series([None] * len(df), dtype=pl.Utf8),
        "target_seq": [target_seq] * len(df),
        "kd_nm": df["KD (nM)"].cast(pl.Float64),
        "source": ["shanehsazzadeh"] * len(df),
        "antibody": ["de_novo_HER2_binder"] * len(df),
        "antigen": ["HER2"] * len(df),
        "notes": ["HCDR3 only; full VH/VL unavailable"] * len(df),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run curation pipeline for all datasets."""
    print("Curating DMS datasets...")

    curators = [
        ("phillips", curate_phillips),
        ("dailey", curate_dailey),
        ("magma_seq", curate_magma_seq),
        ("engelhart", curate_engelhart),
        ("shanehsazzadeh", curate_shanehsazzadeh),
    ]

    all_frames: list[pl.DataFrame] = []
    for name, fn in curators:
        out_path = OUT_DIR / f"{name}_curated.csv"
        if out_path.exists():
            print(f"  {name}: {out_path.name} already exists, loading cached version")
            df = pl.read_csv(out_path)
        else:
            print(f"\n  Curating {name}...")
            df = fn()
            df.write_csv(out_path)
            print(f"  {name}: {len(df):,} rows → {out_path.name}")
        all_frames.append(df)

    # Merge
    merged = pl.concat(all_frames)
    merged_path = OUT_DIR / "all_data.csv"
    merged.write_csv(merged_path)
    print(f"\nAll datasets merged: {len(merged):,} rows → {merged_path.name}")

    # Summary
    print("\nRow counts by source:")
    print(merged.group_by("source").agg(pl.len().alias("n")).sort("n", descending=True))

    print("\nKD range by dataset:")
    print(
        merged.group_by("source")
        .agg(
            pl.col("kd_nm").min().alias("kd_min_nM"),
            pl.col("kd_nm").median().alias("kd_median_nM"),
            pl.col("kd_nm").max().alias("kd_max_nM"),
        )
        .sort("source")
    )


if __name__ == "__main__":
    main()
