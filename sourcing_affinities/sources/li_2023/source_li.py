"""Source script: Li et al. 2023 ML-generated aayl49_ML scFv variants.

Note: aayl49_ML_benchmarking_data.csv may already exist in DMS_Data/05_engelhart_aayl/
and be processed by source_engelhart.py (which includes it in TARGET_FILES).
Check whether that file is already covered before adding this source separately.

If the Li 2023 data is separate from the Engelhart 2022 source files, place raw
files in raw/ and implement the parsing below.

STUB — see README.md for download instructions.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sourcing_affinities"))

SOURCE_KEY = "li_2023"
RAW_DIR = Path(__file__).parent / "raw"
OUT_FILE = Path(__file__).parent / "li_2023.parquet"

CITATION = (
    "Li et al. (2023) Machine learning optimization of candidate antibody "
    "yields high-affinity anti-VEGF antibodies. [confirm full citation]"
)
METHOD = "phage_display_alphaSeq"
GROSS_ANTIGEN = "VEGF"
FINE_ANTIGEN = "VEGF-A"

_ENGELHART_CSV = (
    REPO_ROOT / "DMS_Data" / "05_engelhart_aayl" / "aayl49_ML_benchmarking_data.csv"
)


def source_li():
    # Check if already covered by Engelhart source
    if _ENGELHART_CSV.exists():
        print(
            f"[{SOURCE_KEY}] aayl49_ML data already at {_ENGELHART_CSV}.\n"
            "It is included in source_engelhart.py (TARGET_FILES). "
            "No separate processing needed unless citation should differ."
        )
        return None

    raw_files = list(RAW_DIR.glob("*"))
    if not raw_files:
        raise FileNotFoundError(
            f"\n[{SOURCE_KEY}] No raw data found in {RAW_DIR}.\n"
            "See README.md for download instructions."
        )
    raise NotImplementedError("source_li: implement after downloading raw data.")


if __name__ == "__main__":
    result = source_li()
    if result is not None:
        print(result.head(3))
