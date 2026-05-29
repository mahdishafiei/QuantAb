"""
Extract IgBERT and/or ESM-2 embeddings for all DMS datasets and save to disk.

Usage
-----
# Both models, default settings
python scripts/extract_embeddings.py --data-dir /path/to/DMS_Data

# Single model, 10 PCA components
python scripts/extract_embeddings.py --data-dir /path/to/DMS_Data --model igbert --n-components 10

# Skip already-computed files (idempotent by default)
python scripts/extract_embeddings.py --data-dir /path/to/DMS_Data --overwrite

Output structure
----------------
embeddings/
  igbert/
    phillips_raw.npy       # raw 1024-dim embeddings (N, 1024)
    phillips_6d.npy        # PCA-reduced (N, 6)
    phillips_10d.npy       # PCA-reduced (N, 10)
    engelhart_raw.npy
    ...
  esm2_35M/
    phillips_raw.npy       # raw 480-dim embeddings
    ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from quantab.data import load_all, summarize
from quantab.embeddings import REGISTRY, extract_embeddings, fit_pca, load_model

PCA_DIMS = [6, 10]
BATCH_SIZE = 64


def get_sequences(df: pl.DataFrame, use_paired: bool = True) -> list[str]:
    if use_paired and "light" in df.columns:
        return [
            h + l if l is not None else h
            for h, l in zip(df["heavy"].to_list(), df["light"].to_list())
        ]
    return df["heavy"].to_list()


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    models = list(REGISTRY.keys()) if args.model == "all" else [args.model]

    print(f"\n{'='*60}")
    print(f"Loading datasets from {data_dir}")
    df_all = load_all(data_dir)
    summarize(df_all)

    # Group by dataset name
    datasets: dict[str, pl.DataFrame] = {
        name: df_all.filter(pl.col("dataset") == name)
        for name in df_all["dataset"].unique().to_list()
    }
    # Add a combined split for the main Phillips datasets
    phillips_df = df_all.filter(pl.col("dataset").str.starts_with("phillips"))
    if len(phillips_df) > 0:
        datasets["phillips_all"] = phillips_df

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_key}  ({REGISTRY[model_key].hf_name})")
        print(f"{'='*60}")

        tok, model, cfg = load_model(model_key)
        model_dir = out_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        for ds_name, df in datasets.items():
            raw_path = model_dir / f"{ds_name}_raw.npy"

            # --- Raw embeddings ---
            if raw_path.exists() and not args.overwrite:
                print(f"  [{ds_name}] raw embeddings exist, loading from cache.")
                raw = np.load(raw_path)
            else:
                print(f"  [{ds_name}] extracting embeddings for {len(df)} sequences...")
                seqs = get_sequences(df, use_paired=True)
                raw = extract_embeddings(seqs, tok, model, cfg, batch_size=BATCH_SIZE)
                np.save(raw_path, raw)
                print(f"  [{ds_name}] saved raw embeddings → {raw_path}  shape={raw.shape}")

            # --- PCA reductions ---
            for n in args.n_components:
                pca_path = model_dir / f"{ds_name}_{n}d.npy"
                if pca_path.exists() and not args.overwrite:
                    print(f"  [{ds_name}] {n}d PCA exists, skipping.")
                    continue
                pca, X = fit_pca(raw, n_components=n)
                np.save(pca_path, X)
                var = pca.explained_variance_ratio_.sum()
                print(f"  [{ds_name}] PCA {n}d → {pca_path}  explained_var={var:.3f}")

            # --- Save labels alongside embeddings (once per dataset) ---
            labels_path = model_dir / f"{ds_name}_labels.npy"
            if not labels_path.exists() or args.overwrite:
                y = df["affinity"].to_numpy().astype("float32")
                np.save(labels_path, y)

        # Save model metadata
        meta_path = model_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "model_key": model_key,
                    "hf_name": cfg.hf_name,
                    "hidden_dim": cfg.hidden_dim,
                    "pool": cfg.pool,
                    "pca_dims": args.n_components,
                    "datasets": list(datasets.keys()),
                },
                f,
                indent=2,
            )

    print(f"\nDone. Embeddings saved to {out_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract AbLM embeddings for all DMS datasets.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to DMS_Data directory containing dataset subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default="embeddings",
        help="Directory to save embeddings (default: embeddings/).",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "igbert", "esm2_35M", "esm2_150M", "esm2_650M", "balm_paired"],
        help="Which model(s) to run (default: all).",
    )
    parser.add_argument(
        "--n-components",
        nargs="+",
        type=int,
        default=PCA_DIMS,
        help=f"PCA component counts to save (default: {PCA_DIMS}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and overwrite existing embedding files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
