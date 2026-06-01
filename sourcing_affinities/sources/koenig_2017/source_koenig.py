"""Source script: Koenig et al. 2017 Ranibizumab anti-VEGF DMS.

STUB — data not yet downloaded. Place raw files in raw/ and implement parsing.

Paper: Koenig P et al. (2017) Mutational landscape of antibody variable domains
       reveals a switch between stability and binding optimization at cold
       temperatures. PNAS 114(3):497-502. doi:10.1073/pnas.1613231114

PDB: 2FJG — Ranibizumab Fab (chains H, L) bound to VEGF-A fragment (chain V).
Method: phage display DMS.
Target: VEGF-A (use chain V from PDB 2FJG — fetch with: wget
        https://files.rcsb.org/download/2FJG.pdb -O raw/2FJG.pdb)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sourcing_affinities"))

SOURCE_KEY = "koenig_2017"
RAW_DIR = Path(__file__).parent / "raw"
OUT_FILE = Path(__file__).parent / "koenig_2017.parquet"

CITATION = (
    "Koenig P et al. (2017) Mutational landscape of antibody variable domains "
    "reveals a switch between stability and binding optimization at cold temperatures. "
    "PNAS 114(3):497-502. doi:10.1073/pnas.1613231114"
)
METHOD = "phage_display"
GROSS_ANTIGEN = "VEGF"
FINE_ANTIGEN = "VEGF-A"


def source_koenig():
    raw_files = list(RAW_DIR.glob("*"))
    if not raw_files:
        raise FileNotFoundError(
            f"\n[{SOURCE_KEY}] No raw data found in {RAW_DIR}.\n"
            "See README.md for download instructions."
        )
    # TODO: implement parsing once raw data is available.
    # Expected steps:
    # 1. Load mutation CSV/Excel from raw/
    # 2. Fetch PDB 2FJG if not present: raw/2FJG.pdb
    # 3. Extract WT VH (chain H), VL (chain L), VEGF-A (chain V) from SEQRES
    # 4. Apply mutations to reconstruct variant sequences
    # 5. Convert affinity (enrichment or KD) to kd_nm
    # 6. apply_censoring, validate_df, cast_to_schema
    raise NotImplementedError("source_koenig: implement after downloading raw data.")


if __name__ == "__main__":
    source_koenig()
