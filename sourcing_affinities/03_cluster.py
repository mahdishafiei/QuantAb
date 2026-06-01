"""Pipeline step 3: Cluster antibody and antigen sequences with MMseqs2.

HC (antibody heavy chain) thresholds:  40, 70, 90, 95, 97 % identity
AG (antigen) thresholds:                80, 95 % identity

Adds per-sequence columns:
  hc_cluster_{thr}   — cluster representative ID at that threshold
  hc_weight_{thr}    — 1 / cluster_size (inverse frequency weight)
  ag_cluster_{thr}
  ag_weight_{thr}

Then creates all 10 permutation joint-cluster columns:
  joint_ab{X}_ag{Y}        — joint cluster ID (hc_rep + "__" + ag_rep)
  joint_weight_ab{X}_ag{Y} — 1 / n_rows sharing that joint cluster

Reads  outputs/affinity_corpus.parquet
Writes outputs/affinity_corpus.parquet     (updated in-place with cluster cols)
       outputs/cluster_assignments.parquet (cluster cols + key sequences only)

Usage:
    python sourcing_affinities/03_cluster.py [--filtered-only]
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MMSEQS_BIN = Path(
    "/home/jovyan/work/Software/abutils/abutils/binaries/mmseqs_linux_amd64"
)

HC_THRESHOLDS = [40, 70, 90, 95, 97]   # antibody heavy chain
AG_THRESHOLDS = [80, 95]               # antigen

SOURCING_DIR = Path(__file__).parent
OUT_DIR = SOURCING_DIR / "outputs"
CORPUS_PATH = OUT_DIR / "affinity_corpus.parquet"
CLUSTER_PATH = OUT_DIR / "cluster_assignments.parquet"

N_THREADS = min(8, os.cpu_count() or 4)


# ---------------------------------------------------------------------------
# MMseqs2 helpers
# ---------------------------------------------------------------------------

def _write_fasta(sequences: list[str], ids: list[str], path: Path) -> None:
    with open(path, "w") as fh:
        for seq_id, seq in zip(ids, sequences):
            fh.write(f">{seq_id}\n{seq}\n")


def _parse_cluster_tsv(tsv_path: Path) -> dict[str, str]:
    """Parse MMseqs2 *_cluster.tsv (rep TAB member) → {member: rep} dict."""
    m2c: dict[str, str] = {}
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 2:
                m2c[parts[1]] = parts[0]
    return m2c


def run_mmseqs_cluster(
    sequences: list[str],
    ids: list[str],
    threshold: int,
    tmpdir: Path,
    tag: str = "",
) -> dict[str, str]:
    """Cluster sequences at `threshold`% identity. Returns {seq_id: cluster_rep}."""
    thr_frac = threshold / 100.0
    fasta = tmpdir / f"input_{tag}_{threshold}.fasta"
    prefix = tmpdir / f"cluster_{tag}_{threshold}"
    work = tmpdir / f"tmp_{tag}_{threshold}"
    work.mkdir(exist_ok=True)

    _write_fasta(sequences, ids, fasta)

    cmd = [
        str(MMSEQS_BIN), "easy-cluster",
        str(fasta), str(prefix), str(work),
        "--min-seq-id", str(thr_frac),
        "--cov-mode", "0",
        "-c", "0.8",
        "--threads", str(N_THREADS),
        "-v", "1",               # suppress verbose output
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"MMseqs2 failed (threshold={threshold}%, tag={tag}):\n"
            + result.stderr[-600:]
        )

    tsv = Path(str(prefix) + "_cluster.tsv")
    if not tsv.exists():
        raise FileNotFoundError(f"MMseqs2 output not found: {tsv}")

    return _parse_cluster_tsv(tsv)


# ---------------------------------------------------------------------------
# Clustering pipeline
# ---------------------------------------------------------------------------

def cluster_column(
    df: pl.DataFrame,
    seq_col: str,
    thresholds: list[int],
    prefix: str,
    tmpdir: Path,
) -> pl.DataFrame:
    """Run MMseqs2 at each threshold and add cluster + weight columns to df.

    Only the *unique* sequences in `seq_col` are sent to MMseqs2; cluster
    representatives are then mapped back to every row via a join. This keeps
    MMseqs2 input small (the corpus has heavy sequence redundancy) without
    changing results — cluster sizes/weights are still counted over all rows.
    """
    unique_seqs = df[seq_col].unique(maintain_order=True).to_list()
    ids = [f"seq_{i}" for i in range(len(unique_seqs))]
    id_to_seq = dict(zip(ids, unique_seqs))
    print(f"  {prefix}: {len(unique_seqs):,} unique / {len(df):,} rows "
          f"({len(df) / max(len(unique_seqs), 1):.1f}x redundancy)")

    for thr in tqdm(thresholds, desc=f"{prefix} clustering", leave=True):
        col_cluster = f"{prefix}_cluster_{thr}"
        col_weight = f"{prefix}_weight_{thr}"

        m2c = run_mmseqs_cluster(unique_seqs, ids, thr, tmpdir, tag=prefix)

        # member-id -> rep-id  becomes  sequence-string -> rep-id
        seq_to_rep = {
            seq: m2c.get(sid, sid) for sid, seq in id_to_seq.items()
        }
        mapping = pl.DataFrame(
            {seq_col: list(seq_to_rep.keys()),
             col_cluster: list(seq_to_rep.values())}
        )
        df = df.join(mapping, on=seq_col, how="left")

        sizes = df.group_by(col_cluster).len().rename({"len": "_sz"})
        df = df.join(sizes, on=col_cluster, how="left")
        df = df.with_columns((1.0 / pl.col("_sz")).alias(col_weight)).drop("_sz")

        n_clusters = df[col_cluster].n_unique()
        med_size = df.group_by(col_cluster).len()["len"].median()
        max_size = df.group_by(col_cluster).len()["len"].max()
        print(
            f"  {prefix}@{thr}%: {n_clusters:,} clusters | "
            f"median size={med_size:.1f} | max size={max_size:,}"
        )

    return df


def add_joint_clusters(df: pl.DataFrame) -> pl.DataFrame:
    """Create joint cluster columns for every (hc_thr, ag_thr) permutation."""
    for hc_t, ag_t in itertools.product(HC_THRESHOLDS, AG_THRESHOLDS):
        hc_col = f"hc_cluster_{hc_t}"
        ag_col = f"ag_cluster_{ag_t}"
        joint_col = f"joint_ab{hc_t}_ag{ag_t}"
        weight_col = f"joint_weight_ab{hc_t}_ag{ag_t}"

        if hc_col not in df.columns or ag_col not in df.columns:
            continue

        df = df.with_columns(
            (pl.col(hc_col) + "__" + pl.col(ag_col)).alias(joint_col)
        )

        # Weight = 1 / joint cluster size
        sizes = df.group_by(joint_col).len().rename({"len": "_jsz"})
        df = df.join(sizes, on=joint_col, how="left")
        df = df.with_columns((1.0 / pl.col("_jsz")).alias(weight_col)).drop("_jsz")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filtered-only", action="store_true",
        help="Cluster only non-censored rows (faster for large datasets).",
    )
    args = parser.parse_args()

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}\nRun 02_aggregate.py first."
        )
    if not MMSEQS_BIN.exists():
        raise FileNotFoundError(f"MMseqs2 binary not found: {MMSEQS_BIN}")

    df = pl.read_parquet(CORPUS_PATH)
    if args.filtered_only:
        df = df.filter(~pl.col("is_censored"))
    print(f"\n[03_cluster] {len(df):,} rows | "
          f"HC thresholds: {HC_THRESHOLDS} | AG thresholds: {AG_THRESHOLDS}")

    with tempfile.TemporaryDirectory(prefix="quantab_cluster_") as tmpstr:
        tmpdir = Path(tmpstr)

        print("\n--- Antibody heavy chain (hc_seq) ---")
        df = cluster_column(df, "hc_seq", HC_THRESHOLDS, "hc", tmpdir)

        print("\n--- Antigen (target_seq) ---")
        df = cluster_column(df, "target_seq", AG_THRESHOLDS, "ag", tmpdir)

    print("\n--- Building joint cluster permutations ---")
    df = add_joint_clusters(df)

    combos = list(itertools.product(HC_THRESHOLDS, AG_THRESHOLDS))
    print(f"  {len(combos)} permutations: "
          + ", ".join(f"ab{h}_ag{a}" for h, a in combos))

    for hc_t, ag_t in combos:
        jcol = f"joint_ab{hc_t}_ag{ag_t}"
        if jcol in df.columns:
            n = df[jcol].n_unique()
            max_sz = df.group_by(jcol).len()["len"].max()
            print(f"  {jcol}: {n:,} joint clusters | max size={max_sz:,}")

    # Save cluster assignments (compact: keys + all cluster/weight cols)
    key_cols = ["hc_seq", "lc_seq", "target_seq", "source_key"]
    cluster_cols = (
        [f"hc_cluster_{t}" for t in HC_THRESHOLDS]
        + [f"hc_weight_{t}" for t in HC_THRESHOLDS]
        + [f"ag_cluster_{t}" for t in AG_THRESHOLDS]
        + [f"ag_weight_{t}" for t in AG_THRESHOLDS]
        + [f"joint_ab{h}_ag{a}" for h, a in combos]
        + [f"joint_weight_ab{h}_ag{a}" for h, a in combos]
    )
    keep = [c for c in key_cols + cluster_cols if c in df.columns]
    df.select(keep).write_parquet(CLUSTER_PATH)
    print(f"\nCluster assignments → {CLUSTER_PATH}")

    # Update corpus in-place
    df.write_parquet(CORPUS_PATH)
    print(f"Corpus updated      → {CORPUS_PATH}")


if __name__ == "__main__":
    main()
