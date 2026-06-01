"""Pipeline step 1: Validate and clean all available source parquets.

Runs each source script (for sources with raw data), applies validate_df(),
and prints a summary table of rows kept vs. dropped per source.

Usage:
    python sourcing_affinities/01_clean_sources.py [--run-sources]

By default reads existing *.parquet files from sources/*/. Use --run-sources
to re-run each source_*.py script (re-processes raw data).
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import polars as pl

SOURCING_DIR = Path(__file__).parent
SOURCES_DIR = SOURCING_DIR / "sources"
sys.path.insert(0, str(SOURCING_DIR))

from schema import CENSORING_THRESHOLD_NM, apply_censoring, cast_to_schema, validate_df  # noqa: E402

# Sources in the order to process (stubs will be skipped gracefully)
SOURCE_SCRIPTS: list[tuple[str, str]] = [
    ("phillips_2021",       "sources/phillips_2021/source_phillips.py"),
    ("engelhart_2022",      "sources/engelhart_2022/source_engelhart.py"),
    ("shanehsazzadeh_2023", "sources/shanehsazzadeh_2023/source_shanehsazzadeh.py"),
    ("andd_2026",           "sources/andd_2026/source_andd.py"),
    ("dailey_2021",         "sources/dailey_2021/source_dailey.py"),
    ("magma_seq_2022",      "sources/magma_seq_2022/source_magma_seq.py"),
    ("koenig_2017",         "sources/koenig_2017/source_koenig.py"),
    ("warszawski_2020",     "sources/warszawski_2020/source_warszawski.py"),
    ("clark_2006",          "sources/clark_2006/source_clark.py"),
    ("li_2023",             "sources/li_2023/source_li.py"),
    ("minot_2024",          "sources/minot_2024/source_minot.py"),
    ("cov_abdab_unibind",   "sources/cov_abdab_unibind/source_cov_abdab.py"),
]


def run_source_script(source_key: str, script_path: Path) -> bool:
    """Run a source script as a subprocess. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"Running: {script_path.name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  [SKIP] {source_key}: script exited with code {result.returncode}")
        return False
    return True


def load_and_validate_parquet(parquet_path: Path, source_key: str) -> pl.DataFrame | None:
    """Load a source parquet, re-run validation, and return the cleaned frame."""
    if not parquet_path.exists():
        return None
    df = pl.read_parquet(parquet_path)
    df = validate_df(df, source_key=source_key)
    df = cast_to_schema(df)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-sources", action="store_true",
        help="Re-run source_*.py scripts to regenerate parquets from raw data."
    )
    args = parser.parse_args()

    summary_rows = []
    all_dfs: list[pl.DataFrame] = []

    for source_key, script_rel in SOURCE_SCRIPTS:
        script_path = SOURCING_DIR / script_rel
        parquet_path = script_path.parent / f"{source_key}.parquet"

        if args.run_sources:
            success = run_source_script(source_key, script_path)
            if not success:
                summary_rows.append({
                    "source": source_key, "status": "SKIP/ERROR",
                    "n_raw": 0, "n_clean": 0, "n_censored": 0,
                })
                continue

        df = load_and_validate_parquet(parquet_path, source_key)
        if df is None:
            print(f"  [SKIP] {source_key}: no parquet at {parquet_path}")
            summary_rows.append({
                "source": source_key, "status": "MISSING",
                "n_raw": 0, "n_clean": 0, "n_censored": 0,
            })
            continue

        n_censored = df.filter(pl.col("is_censored")).height
        summary_rows.append({
            "source": source_key,
            "status": "OK",
            "n_raw": len(df),
            "n_clean": len(df),
            "n_censored": n_censored,
        })
        all_dfs.append(df)

        # Write back the validated parquet
        df.write_parquet(parquet_path)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    summary = pl.DataFrame(summary_rows)
    print(summary)
    total_clean = summary.filter(pl.col("status") == "OK")["n_clean"].sum()
    total_censored = summary.filter(pl.col("status") == "OK")["n_censored"].sum()
    print(f"\nTotal rows (all OK sources): {total_clean:,}")
    print(f"Total censored:              {total_censored:,}")
    print(f"Total non-censored:          {total_clean - total_censored:,}")


if __name__ == "__main__":
    main()
