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

from quantab.data import load_all, load_curated, load_unified, summarize
from quantab.embeddings import REGISTRY, ModelConfig, extract_embeddings, fit_pca, load_model

PCA_DIMS = [6, 10]
BATCH_SIZE = 64


def get_sequences(df: pl.DataFrame, cfg: ModelConfig, embed: str = "antibody"):
    """Build the model's expected input sequences.

    embed="antibody": antibody chains.
      - Paired models (balm_paired): list of ``(heavy, light)`` tuples (native pairing).
      - Single-chain models: heavy + light concatenated into one string when light exists.
    embed="target": the antigen sequence (single string per row) — for embedding the
      target/antigen modality with a general protein model (ESM-2).
    """
    if embed == "target":
        seqs = df["target"].to_list()
        return [(s, "") for s in seqs] if cfg.paired else seqs

    heavy = df["heavy"].to_list()
    light = df["light"].to_list() if "light" in df.columns else [None] * len(heavy)
    if cfg.paired:
        return [(h, (l or "")) for h, l in zip(heavy, light)]
    if "light" in df.columns:
        return [h + l if l else h for h, l in zip(heavy, light)]
    return heavy


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    models = list(REGISTRY.keys()) if args.model == "all" else [args.model]

    print(f"\n{'='*60}")
    if args.unified:
        # Prebuilt unified affinity dataset (heavy/light/target/affinity/affinity_norm)
        print(f"Loading unified dataset from {args.unified}  (embed={args.embed})")
        df_all = load_unified(Path(args.unified))
        summarize(df_all)
        datasets: dict[str, pl.DataFrame] = {"unified": df_all}
    elif args.data_file:
        # Curated single-file dataset (filtered_data.csv from S3)
        print(f"Loading curated dataset from {args.data_file}")
        df_all = load_curated(Path(args.data_file))
        summarize(df_all)
        datasets = {"curated": df_all}
    else:
        data_dir = Path(args.data_dir)
        print(f"Loading datasets from {data_dir}")
        df_all = load_all(data_dir)
        summarize(df_all)
        # Group by dataset name
        datasets = {
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
            # antigen embeddings get a distinct prefix so they sit beside antibody ones
            tag = f"{ds_name}_antigen" if args.embed == "target" else ds_name
            raw_path = model_dir / f"{tag}_raw.npy"

            # --- Raw embeddings ---
            if raw_path.exists() and not args.overwrite:
                print(f"  [{tag}] raw embeddings exist, loading from cache.")
                raw = np.load(raw_path)
            else:
                print(f"  [{tag}] extracting embeddings for {len(df)} sequences (embed={args.embed}, max_length={args.max_length})...")
                seqs = get_sequences(df, cfg, embed=args.embed)
                raw = extract_embeddings(seqs, tok, model, cfg, batch_size=BATCH_SIZE,
                                         max_length=args.max_length)
                np.save(raw_path, raw)
                print(f"  [{tag}] saved raw embeddings → {raw_path}  shape={raw.shape}")

            # --- PCA reductions ---
            for n in args.n_components:
                pca_path = model_dir / f"{tag}_{n}d.npy"
                if pca_path.exists() and not args.overwrite:
                    print(f"  [{tag}] {n}d PCA exists, skipping.")
                    continue
                pca, X = fit_pca(raw, n_components=n)
                np.save(pca_path, X)
                var = pca.explained_variance_ratio_.sum()
                print(f"  [{tag}] PCA {n}d → {pca_path}  explained_var={var:.3f}")

            # --- Save labels alongside embeddings (raw pKd + normalized) ---
            if args.embed != "target":
                labels_path = model_dir / f"{ds_name}_labels.npy"
                if (not labels_path.exists() or args.overwrite) and "affinity" in df.columns:
                    np.save(labels_path, df["affinity"].to_numpy().astype("float32"))
                norm_path = model_dir / f"{ds_name}_labels_norm.npy"
                if (not norm_path.exists() or args.overwrite) and "affinity_norm" in df.columns:
                    np.save(norm_path, df["affinity_norm"].to_numpy().astype("float32"))

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
        default=None,
        help="Path to DMS_Data directory containing dataset subdirectories.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Path to the curated single-file dataset (e.g. curated_data/filtered_data.csv). "
        "Takes precedence over --data-dir.",
    )
    parser.add_argument(
        "--unified",
        default=None,
        help="Path to the prebuilt unified affinity dataset (unified_affinity.csv). "
        "Takes precedence over --data-file / --data-dir.",
    )
    parser.add_argument(
        "--embed",
        default="antibody",
        choices=["antibody", "target"],
        help="What to embed: 'antibody' (heavy[+light]) or 'target' (antigen sequence).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max tokens. Antibody fits in 512; use 1024 for antigen (ESM-2 context).",
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
    args = parser.parse_args()
    if not args.unified and not args.data_file and not args.data_dir:
        parser.error("provide --unified, --data-file, or --data-dir")
    return args


if __name__ == "__main__":
    run(parse_args())
