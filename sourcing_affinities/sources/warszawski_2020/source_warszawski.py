"""Source script: Warszawski et al. 2020 anti-VEGF yeast display DMS.

STUB — data not yet downloaded. Place raw files in raw/ and implement parsing.

Paper: Warszawski S et al. (2020) Optimizing antibody affinity and stability
       by the automated design of the variable light-heavy chain interfaces.
       PLOS Comput Biol 16(4):e1007828. doi:10.1371/journal.pcbi.1007828

PDB: 1MLC — Fab (chains B=VH, A=VL) bound to VEGF-A (chain E).
Method: yeast display DMS.
Target: VEGF-A (fetch PDB: wget https://files.rcsb.org/download/1MLC.pdb -O raw/1MLC.pdb)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sourcing_affinities"))

SOURCE_KEY = "warszawski_2020"
RAW_DIR = Path(__file__).parent / "raw"
OUT_FILE = Path(__file__).parent / "warszawski_2020.parquet"

CITATION = (
    "Warszawski S et al. (2020) Optimizing antibody affinity and stability by the "
    "automated design of the variable light-heavy chain interfaces. "
    "PLOS Comput Biol 16(4):e1007828. doi:10.1371/journal.pcbi.1007828"
)
METHOD = "yeast_display"
GROSS_ANTIGEN = "VEGF"
FINE_ANTIGEN = "VEGF-A"


def source_warszawski():
    raw_files = list(RAW_DIR.glob("*"))
    if not raw_files:
        raise FileNotFoundError(
            f"\n[{SOURCE_KEY}] No raw data found in {RAW_DIR}.\n"
            "See README.md for download instructions."
        )
    raise NotImplementedError("source_warszawski: implement after downloading raw data.")


if __name__ == "__main__":
    source_warszawski()
