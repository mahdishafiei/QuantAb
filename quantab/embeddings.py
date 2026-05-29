"""Antibody language model embedding extraction and PCA dimensionality reduction.

Supported models
----------------
igbert       : Exscientia/IgBert              — antibody-specific BERT, 1024-dim
esm2_35M     : facebook/esm2_t12_35M_UR50D   — general protein ESM-2,  480-dim
esm2_150M    : facebook/esm2_t30_150M_UR50D  — general protein ESM-2,  640-dim
esm2_650M    : facebook/esm2_t33_650M_UR50D  — general protein ESM-2, 1280-dim  ← recommended
balm_paired  : brineylab/BALM-paired          — paired antibody RoBERTa, 1024-dim (Burbach & Briney 2024)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import polars as pl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelConfig:
    hf_name: str
    hidden_dim: int
    space_sep: bool   # True → "E V Q L...", False → raw sequence
    pool: str         # "cls" or "mean"
    paired: bool = False  # True → tokenizer receives (heavy, light) as two sequences


REGISTRY: dict[str, ModelConfig] = {
    "igbert": ModelConfig(
        hf_name="Exscientia/IgBert",
        hidden_dim=1024,
        space_sep=True,
        pool="cls",
        paired=False,
    ),
    "esm2_35M": ModelConfig(
        hf_name="facebook/esm2_t12_35M_UR50D",
        hidden_dim=480,
        space_sep=False,
        pool="mean",
        paired=False,
    ),
    "esm2_150M": ModelConfig(
        hf_name="facebook/esm2_t30_150M_UR50D",
        hidden_dim=640,
        space_sep=False,
        pool="mean",
        paired=False,
    ),
    "esm2_650M": ModelConfig(
        hf_name="facebook/esm2_t33_650M_UR50D",
        hidden_dim=1280,
        space_sep=False,
        pool="mean",
        paired=False,
    ),
    "balm_paired": ModelConfig(
        hf_name="brineylab/BALM-paired",
        hidden_dim=1024,
        space_sep=False,
        pool="cls",
        paired=True,   # passes heavy and light as two sequences to the tokenizer
    ),
}


def load_model(model_key: str, device: str = DEVICE):
    """Load tokenizer and model by registry key.

    Args:
        model_key: One of "igbert", "esm2_35M", "esm2_150M", "esm2_650M", "balm_paired".
        device: torch device string.

    Returns:
        (tokenizer, model, config)
    """
    if model_key not in REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(REGISTRY)}")
    cfg = REGISTRY[model_key]
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_name)
    model = AutoModel.from_pretrained(cfg.hf_name).to(device).eval()
    return tokenizer, model, cfg


def _prepare(seq: str, space_sep: bool) -> str:
    return " ".join(seq.strip()) if space_sep else seq.strip()


def _pool(hidden: torch.Tensor, attention_mask: torch.Tensor, strategy: str) -> np.ndarray:
    if strategy == "cls":
        return hidden[:, 0, :].cpu().float().numpy()
    mask = attention_mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).cpu().float().numpy()


@torch.no_grad()
def extract_embeddings(
    sequences: list[str] | list[tuple[str, str]],
    tokenizer,
    model,
    cfg: ModelConfig,
    batch_size: int = 32,
    device: str = DEVICE,
    max_length: int = 512,
) -> np.ndarray:
    """Extract embeddings for a list of sequences.

    Args:
        sequences: For single-chain models: list of amino acid strings.
                   For paired models (balm_paired): list of (heavy, light) tuples.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model.
        cfg: ModelConfig for this model.
        batch_size: Sequences per forward pass.
        device: torch device string.
        max_length: Maximum token length (truncates longer sequences).

    Returns:
        Float32 array of shape (N, hidden_dim).
    """
    all_embeddings: list[np.ndarray] = []
    label = cfg.hf_name.split("/")[-1]

    for i in tqdm(range(0, len(sequences), batch_size), desc=f"Extracting ({label})"):
        batch = sequences[i : i + batch_size]

        if cfg.paired:
            # BALM-paired: tokenizer takes (heavy_list, light_list) as two sequences
            heavy_batch = [_prepare(h, cfg.space_sep) for h, _ in batch]
            light_batch = [_prepare(l, cfg.space_sep) for _, l in batch]
            enc = tokenizer(
                heavy_batch,
                light_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
        else:
            enc = tokenizer(
                [_prepare(s, cfg.space_sep) for s in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

        out = model(**enc)
        emb = _pool(out.last_hidden_state, enc["attention_mask"], cfg.pool)
        all_embeddings.append(emb)

    return np.vstack(all_embeddings)


def fit_pca(embeddings: np.ndarray, n_components: int) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return pca, reduced


def embed_and_reduce(
    df: pl.DataFrame,
    tokenizer,
    model,
    cfg: ModelConfig,
    n_components: int = 10,
    batch_size: int = 32,
    cache_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Full pipeline: sequences → embeddings → PCA → (X, y).

    For paired models (balm_paired), heavy and light chains are passed
    separately to the tokenizer. For all others, heavy+light are concatenated
    into a single string if light is available.

    Args:
        df: DataFrame with columns heavy, light (optional), affinity.
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model.
        cfg: ModelConfig for this model.
        n_components: PCA output dimensions (= n_qubits).
        batch_size: Forward pass batch size.
        cache_path: Save/load raw embeddings to avoid re-running.

    Returns:
        X: PCA-reduced features (N, n_components).
        y: Affinity values (N,).
        pca: Fitted PCA object.
    """
    y = df["affinity"].to_numpy().astype(np.float32)

    if cache_path is not None and cache_path.exists():
        raw = np.load(cache_path)
    else:
        has_light = "light" in df.columns

        if cfg.paired and has_light:
            # Pass (heavy, light) tuples — tokenizer handles pairing
            sequences = [
                (h, l if l is not None else "")
                for h, l in zip(df["heavy"].to_list(), df["light"].to_list())
            ]
        elif has_light:
            # Concatenate for single-chain models
            sequences = [
                h + l if l is not None else h
                for h, l in zip(df["heavy"].to_list(), df["light"].to_list())
            ]
        else:
            sequences = df["heavy"].to_list()

        raw = extract_embeddings(sequences, tokenizer, model, cfg, batch_size=batch_size)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, raw)

    pca, X = fit_pca(raw, n_components=n_components)
    return X, y, pca
