"""Source script: Clark et al. 2006 computationally-directed anti-VEGF mutant library.

STUB — data not yet downloaded. Place raw files in raw/ and implement parsing.

Paper: Clark LA et al. (2006) Affinity enhancement of an in vivo matured
       therapeutic antibody using structure-based computational design.
       Protein Sci 15(5):949-960. doi:10.1110/ps.052030506

PDB: 1MHP — Fab (chains H=VH, L=VL) bound to VEGF-A (chains A, B).
WT sequences already available in DMS_Data/pdbs/1MHP.pdb.
Method: in-silico biophysics-directed mutagenesis; affinities measured by SPR/BLI.
Target: VEGF-A (chain A from 1MHP, 192 aa).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sourcing_affinities"))

SOURCE_KEY = "clark_2006"
RAW_DIR = Path(__file__).parent / "raw"
PDB_1MHP = REPO_ROOT / "DMS_Data" / "pdbs" / "1MHP.pdb"
OUT_FILE = Path(__file__).parent / "clark_2006.parquet"

CITATION = (
    "Clark LA et al. (2006) Affinity enhancement of an in vivo matured therapeutic "
    "antibody using structure-based computational design. "
    "Protein Sci 15(5):949-960. doi:10.1110/ps.052030506"
)
METHOD = "SPR"
GROSS_ANTIGEN = "VEGF"
FINE_ANTIGEN = "VEGF-A"


def source_clark():
    raw_files = [f for f in RAW_DIR.glob("*") if not f.name.endswith(".pdb")]
    if not raw_files:
        raise FileNotFoundError(
            f"\n[{SOURCE_KEY}] No raw data found in {RAW_DIR}.\n"
            "See README.md for download instructions."
        )
    # TODO: implement once raw data available.
    # WT VH from PDB_1MHP chain H (219 aa), VL chain L (212 aa), VEGF chain A (192 aa)
    # Apply mutations, convert KD to nM, run validate_df + cast_to_schema.
    raise NotImplementedError("source_clark: implement after downloading raw data.")


if __name__ == "__main__":
    source_clark()
