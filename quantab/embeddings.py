"""IgBERT embedding extraction and PCA dimensionality reduction."""

from pathlib import Path
from typing import Optional
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
import polars as pl

MODEL_NAME = "Exscientia/IgBert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _space_sep(seq: str) -> str:
    return " ".join(seq.strip())


def load_igbert(device: str = DEVICE):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    return tokenizer, model


@torch.no_grad()
def extract_embeddings(
    sequences: list[str],
    tokenizer,
    model,
    batch_size: int = 32,
    device: str = DEVICE,
    max_length: int = 512,
) -> np.ndarray:
    """Extract [CLS] embeddings from IgBERT for a list of sequences.

    Args:
        sequences: List of amino acid sequences (raw, no spaces needed).
        tokenizer: IgBERT tokenizer.
        model: IgBERT model.
        batch_size: Sequences per forward pass.
        device: torch device string.
        max_length: Max token length (truncates longer sequences).

    Returns:
        Float32 array of shape (N, 768).
    """
    all_embeddings = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
        batch = [_space_sep(s) for s in sequences[i : i + batch_size]]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        out = model(**enc)
        # [CLS] token is index 0
        cls_emb = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        all_embeddings.append(cls_emb)
    return np.vstack(all_embeddings)


def fit_pca(embeddings: np.ndarray, n_components: int) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return pca, reduced


def embed_and_reduce(
    df: pl.DataFrame,
    tokenizer,
    model,
    n_components: int = 10,
    batch_size: int = 32,
    use_paired: bool = True,
    cache_path: Optional[Path] = None,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Full pipeline: sequences → IgBERT embeddings → PCA → (X, y).

    Args:
        df: DataFrame with columns heavy, light (optional), affinity.
        tokenizer: IgBERT tokenizer.
        model: IgBERT model.
        n_components: PCA output dimensions.
        batch_size: IgBERT batch size.
        use_paired: If True and light column exists, concatenate heavy+light.
        cache_path: If provided, save/load raw embeddings from this .npy file.

    Returns:
        X: PCA-reduced features (N, n_components).
        y: Affinity values (N,).
        pca: Fitted PCA object.
    """
    y = df["affinity"].to_numpy().astype(np.float32)

    if cache_path is not None and cache_path.exists():
        raw = np.load(cache_path)
    else:
        if use_paired and "light" in df.columns:
            sequences = [
                h + l if l is not None else h
                for h, l in zip(df["heavy"].to_list(), df["light"].to_list())
            ]
        else:
            sequences = df["heavy"].to_list()

        raw = extract_embeddings(sequences, tokenizer, model, batch_size=batch_size)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, raw)

    pca, X = fit_pca(raw, n_components=n_components)
    return X, y, pca
