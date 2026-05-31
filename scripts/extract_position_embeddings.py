"""
Extract per-position LM embeddings (no pooling) and save as padded arrays.

Unlike extract_embeddings.py, which saves a single pooled vector per sequence,
this script saves the full per-token hidden states so that CNN and BiLSTM
models can learn task-conditioned aggregation.

Output
------
embeddings/<model>/
  unified_pos.npy      float16  (N, max_len, hidden_dim)  — padded to max_len
  unified_lengths.npy  int32    (N,)                      — true sequence lengths

Storage estimate (float16):
  IgBERT (hidden=1024, max_len=256, N=40937): ~21 GB
  ESM2-650M (hidden=1280, max_len=512, N=40937): ~53 GB  ← antigen only if needed

Usage
-----
  # Antibody (IgBERT), unified dataset
  python scripts/extract_position_embeddings.py \\
      --unified curated_data/unified_affinity.csv \\
      --model igbert --max-len 256

  # Dry-run: process only first 500 sequences (sanity check)
  python scripts/extract_position_embeddings.py \\
      --unified curated_data/unified_affinity.csv \\
      --model igbert --max-len 256 --limit 500
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm.auto import tqdm

from quantab.data import load_unified
from quantab.embeddings import REGISTRY, ModelConfig, load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_sequences(df: pl.DataFrame, cfg: ModelConfig) -> list:
    """Build model inputs (same logic as extract_embeddings.py)."""
    heavy = df["heavy"].to_list()
    light = df["light"].to_list() if "light" in df.columns else [None] * len(heavy)
    if cfg.paired:
        return [(h, (l or "")) for h, l in zip(heavy, light)]
    return [h + (l or "") for h, l in zip(heavy, light)]


@torch.no_grad()
def extract_position_embeddings(
    sequences: list,
    tokenizer,
    model,
    cfg: ModelConfig,
    max_len: int,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-position embeddings for all sequences.

    Returns:
        pos_emb: float16 array (N, max_len, hidden_dim), zero-padded.
        lengths:  int32 array  (N,), true token counts (excl. CLS / SEP).
    """
    N = len(sequences)
    hidden_dim = cfg.hidden_dim
    pos_emb = np.zeros((N, max_len, hidden_dim), dtype=np.float16)
    lengths = np.zeros(N, dtype=np.int32)

    label = cfg.hf_name.split("/")[-1]
    for start in tqdm(range(0, N, batch_size), desc=f"Extracting positions ({label})"):
        batch = sequences[start : start + batch_size]

        if cfg.paired:
            heavy_b = [" ".join(h) if cfg.space_sep else h for h, _ in batch]
            light_b = [" ".join(l) if cfg.space_sep else l for _, l in batch]
            enc = tokenizer(
                heavy_b, light_b,
                return_tensors="pt", padding=True,
                truncation=True, max_length=max_len + 2,   # +2 for CLS/SEP
            ).to(DEVICE)
        else:
            seqs = [" ".join(s) if cfg.space_sep else s for s in batch]
            enc = tokenizer(
                seqs,
                return_tensors="pt", padding=True,
                truncation=True, max_length=max_len + 2,
            ).to(DEVICE)

        out = model(**enc)
        hidden = out.last_hidden_state  # (batch, padded_seq, hidden_dim)

        for i, mask_row in enumerate(enc["attention_mask"]):
            # Valid token positions: exclude [CLS] (index 0) and [SEP] (last valid)
            valid_len = int(mask_row.sum().item()) - 2   # subtract CLS + SEP
            valid_len = max(valid_len, 1)
            valid_len = min(valid_len, max_len)

            # Positions 1..valid_len (CLS is 0, sequence starts at 1)
            token_emb = (
                hidden[i, 1 : valid_len + 1, :]
                .cpu()
                .float()
                .numpy()
                .astype(np.float16)
            )
            pos_emb[start + i, :valid_len, :] = token_emb
            lengths[start + i] = valid_len

    return pos_emb, lengths


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)

    print(f"Loading dataset from {args.unified} …")
    df = load_unified(Path(args.unified))
    if args.limit:
        df = df.head(args.limit)
        print(f"  [limit] using first {args.limit} rows.")
    print(f"  N = {len(df)} sequences")

    models = list(REGISTRY.keys()) if args.model == "all" else [args.model]
    # Only antibody-capable models make sense for per-position ab embeddings
    ab_models = {"igbert", "balm_paired", "esm2_35M", "esm2_150M", "esm2_650M"}
    models = [m for m in models if m in ab_models]

    for model_key in models:
        model_dir = out_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        tag = "unified" if not args.limit else f"unified_first{args.limit}"
        pos_path = model_dir / f"{tag}_pos.npy"
        len_path = model_dir / f"{tag}_lengths.npy"

        if pos_path.exists() and len_path.exists() and not args.overwrite:
            print(f"\n[{model_key}] position embeddings exist, skipping.")
            print(f"  {pos_path}  {len_path}")
            continue

        print(f"\n[{model_key}] ({REGISTRY[model_key].hf_name})")
        tok, mdl, cfg = load_model(model_key)

        seqs = get_sequences(df, cfg)
        pos_emb, lengths = extract_position_embeddings(
            seqs, tok, mdl, cfg,
            max_len=args.max_len,
            batch_size=args.batch_size,
        )

        np.save(pos_path, pos_emb)
        np.save(len_path, lengths)
        size_gb = pos_emb.nbytes / 1e9
        print(f"  Saved: {pos_path}  shape={pos_emb.shape}  ({size_gb:.1f} GB)")
        print(f"  Saved: {len_path}  shape={lengths.shape}")
        print(f"  Mean length={lengths.mean():.1f}  max={lengths.max()}  min={lengths.min()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract per-position LM embeddings.")
    p.add_argument("--unified", required=True, help="Path to unified_affinity.csv.")
    p.add_argument("--output-dir", default="embeddings", help="Root embeddings directory.")
    p.add_argument(
        "--model",
        default="igbert",
        choices=["igbert", "balm_paired", "esm2_35M", "esm2_150M", "esm2_650M", "all"],
        help="Which model to extract (default: igbert).",
    )
    p.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Max token positions to store per sequence (default 256). "
             "VH+VL antibodies are typically ~230 tokens; antigen may need 512.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N sequences (sanity check / dry run).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
