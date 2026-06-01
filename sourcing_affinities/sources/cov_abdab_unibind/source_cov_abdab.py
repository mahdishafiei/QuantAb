"""Source script: CoV-AbDab + CoV-UniBind SARS-CoV-2 antibody affinities.

STUB — data not yet downloaded. Place raw files in raw/ and implement parsing.

Sources:
  CoV-AbDab: https://opig.stats.ox.ac.uk/webapps/covabdab/
  CoV-UniBind: bioRxiv 2025.07.17.665346 — quantitative KD table

Strategy: Load CoV-AbDab CSV for HC+LC sequences, join with CoV-UniBind KD
table on antibody identifier. Keep only entries with:
  - Both HC and LC sequences available
  - Quantitative KD (not just IC50/EC50)
  - SARS-CoV-2 as antigen (not just SARS-CoV-1)
  - Known antigen epitope region (RBD, NTD, S2, etc.)

Target sequence: SARS-CoV-2 spike RBD — use 6XC3 chain C from DMS_Data/pdbs/
or fetch Wuhan-Hu-1 reference from UniProt P0DTC2 (aa 319-541 = RBD).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sourcing_affinities"))

SOURCE_KEY = "cov_abdab_unibind"
RAW_DIR = Path(__file__).parent / "raw"
PDB_DIR = REPO_ROOT / "DMS_Data" / "pdbs"
OUT_FILE = Path(__file__).parent / "cov_abdab_unibind.parquet"

CITATION = (
    "CoV-AbDab: Raybould MIJ et al. (2021) Bioinformatics 37(5):734-735. "
    "doi:10.1093/bioinformatics/btaa739. "
    "CoV-UniBind: bioRxiv 2025.07.17.665346. doi:10.1101/2025.07.17.665346"
)
METHOD = "SPR_or_BLI"  # mixed; check Affinity_Method per row
GROSS_ANTIGEN = "SARS-CoV-2"


def source_cov_abdab():
    covabdab_csv = RAW_DIR / "CoV-AbDab.csv"
    unibind_file = next(
        (f for f in RAW_DIR.glob("CoV-UniBind*") if f.suffix in (".csv", ".xlsx")),
        None,
    )

    if not covabdab_csv.exists() or unibind_file is None:
        raise FileNotFoundError(
            f"\n[{SOURCE_KEY}] Missing raw data in {RAW_DIR}.\n"
            "Need: raw/CoV-AbDab.csv AND raw/CoV-UniBind_KD.csv (or .xlsx).\n"
            "See README.md for download instructions."
        )

    # TODO: implement once raw data is downloaded.
    # Steps:
    # 1. Load CoV-AbDab.csv → get HC/LC sequences per antibody name
    # 2. Load CoV-UniBind KD file → get KD (nM), method, antigen per antibody
    # 3. Join on antibody name/ID
    # 4. Filter: Ab_or_Nb == "Ab", has HC+LC, has numeric KD
    # 5. Assign fine_antigen = RBD | NTD | S2 | other from epitope column
    # 6. target_seq = SARS-CoV-2 RBD from 6XC3 chain C (strip His-tag)
    #    or full spike from UniProt P0DTC2 for non-RBD antigens
    # 7. apply_censoring, validate_df, cast_to_schema
    raise NotImplementedError("source_cov_abdab: implement after downloading raw data.")


if __name__ == "__main__":
    source_cov_abdab()
