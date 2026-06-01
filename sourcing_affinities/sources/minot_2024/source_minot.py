"""Source script: Minot & Reddy 2024 anti-Ang2 antibody 5A12 yeast display DMS.

STUB — data not yet downloaded. Place raw files in raw/ and implement parsing.

Paper: Minot M & Reddy ST (2024) Deep mutational scanning of antibody 5A12
       against Angiopoietin-2. [confirm full citation]

Method: yeast display DMS.
Target: Angiopoietin-2 (Ang2) — UniProt O15123 (human, full length 496 aa).
        Fibrinogen-like domain (FLD, aa 284-496) is the antibody binding region.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sourcing_affinities"))

SOURCE_KEY = "minot_2024"
RAW_DIR = Path(__file__).parent / "raw"
OUT_FILE = Path(__file__).parent / "minot_2024.parquet"

CITATION = (
    "Minot M & Reddy ST (2024) Deep mutational scanning of antibody 5A12 "
    "against Angiopoietin-2. [confirm full citation]"
)
METHOD = "yeast_display"
GROSS_ANTIGEN = "Ang2"
FINE_ANTIGEN = "Ang2"


def source_minot():
    raw_files = list(RAW_DIR.glob("*"))
    if not raw_files:
        raise FileNotFoundError(
            f"\n[{SOURCE_KEY}] No raw data found in {RAW_DIR}.\n"
            "See README.md for download instructions."
        )
    # TODO: implement once raw data is available.
    # 1. Load mutation CSV from raw/
    # 2. Fetch Ang2 sequence from UniProt O15123 or use FLD fragment
    # 3. Reconstruct variant VH/VL from WT 5A12 + mutations
    # 4. Convert enrichment/KD to kd_nm
    # 5. apply_censoring, validate_df, cast_to_schema
    raise NotImplementedError("source_minot: implement after downloading raw data.")


if __name__ == "__main__":
    source_minot()
