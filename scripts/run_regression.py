"""Regression benchmark on the model embeddings — leak-free.

Predicts normalized binding affinity (``affinity_norm``) from each model's mean-pooled
embeddings and reports Spearman ρ. Honors the two correctness rules in MEMORY.md:

- ISSUE 1 (PCA leakage): any PCA / scaling lives inside an sklearn Pipeline, so it is
  re-fit on the training split of every CV fold (never on the held-out fold).
- Sequence-similarity leakage: use ``--group-col`` (default ``antigen``) with GroupKFold so
  whole antigens / groups are held out, instead of splitting near-identical DMS single-mutants
  across train and test. Pass ``--group-col none`` for plain random KFold (optimistic).

Reads the embedding layout produced by ``extract_embeddings.py --unified``:
``embeddings/<model>/unified_raw.npy`` + ``unified_labels_norm.npy`` (and, with ``--antigen``,
``embeddings/esm2_650M/unified_antigen_raw.npy``).

Usage
-----
python scripts/run_regression.py --unified curated_data/unified_affinity.csv \
    --model all --antigen --group-col antigen
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANTIBODY_MODELS = ["igbert", "esm2_650M", "balm_paired"]


def build_estimator(pca_dims: int) -> Pipeline:
    """Lab-style MLP head; optional in-fold PCA (fit per training fold → no leak)."""
    steps = [("scaler", StandardScaler())]
    if pca_dims and pca_dims > 0:
        steps.append(("pca", PCA(n_components=pca_dims, random_state=42)))
    steps.append(("mlp", MLPRegressor(
        hidden_layer_sizes=(64, 32), early_stopping=True,
        validation_fraction=0.1, max_iter=200, random_state=42)))
    return Pipeline(steps)


def cv_spearman(X, y, groups, pca_dims, n_splits=5):
    if groups is not None:
        n_splits = min(n_splits, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X)
    scores = []
    for tr, te in splitter:
        est = build_estimator(pca_dims)
        est.fit(X[tr], y[tr])
        scores.append(spearmanr(y[te], est.predict(X[te])).statistic)
    return float(np.mean(scores)), float(np.std(scores)), scores


def run(args: argparse.Namespace) -> None:
    emb = Path(args.embeddings_dir)
    df = pl.read_csv(args.unified, infer_schema_length=5000)
    groups = None
    if args.group_col != "none":
        groups = df[args.group_col].to_numpy()
        print(f"Grouped CV by '{args.group_col}' ({len(np.unique(groups))} groups)")
    else:
        print("Random KFold (no grouping) — optimistic, sequence-similarity leak possible")

    antigen = None
    if args.antigen:
        antigen = np.load(emb / "esm2_650M" / "unified_antigen_raw.npy")
        print(f"Including antigen features (ESM-2): {antigen.shape}")

    models = ANTIBODY_MODELS if args.model == "all" else [args.model]
    results = {}
    print(f"\n{'model':14s} {'antibody':>16s}" + ("   antibody+antigen" if args.antigen else ""))
    for m in models:
        X = np.load(emb / m / "unified_raw.npy")
        y = np.load(emb / m / "unified_labels_norm.npy")
        mean_ab, std_ab, sc_ab = cv_spearman(X, y, groups, args.pca)
        line = f"{m:14s}   {mean_ab:.3f} ± {std_ab:.3f}"
        results[m] = {"antibody": {"mean": mean_ab, "std": std_ab, "scores": sc_ab}}
        if antigen is not None:
            Xj = np.hstack([X, antigen])
            mean_j, std_j, sc_j = cv_spearman(Xj, y, groups, args.pca)
            line += f"     {mean_j:.3f} ± {std_j:.3f}"
            results[m]["antibody_antigen"] = {"mean": mean_j, "std": std_j, "scores": sc_j}
        print(line)

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"group_col": args.group_col, "pca": args.pca, "antigen": args.antigen, "results": results},
        indent=2))
    print(f"\nSaved → {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leak-free regression benchmark on model embeddings.")
    p.add_argument("--embeddings-dir", default="embeddings")
    p.add_argument("--unified", default="curated_data/unified_affinity.csv",
                   help="Unified dataset CSV (for group labels, row-aligned with embeddings).")
    p.add_argument("--model", default="all", choices=["all", *ANTIBODY_MODELS])
    p.add_argument("--antigen", action="store_true", help="Also test antibody+antigen features.")
    p.add_argument("--group-col", default="antigen",
                   choices=["antigen", "norm_group", "source", "none"],
                   help="Grouping column for GroupKFold (default antigen); 'none' = random KFold.")
    p.add_argument("--pca", type=int, default=0,
                   help="In-fold PCA dims (0 = none; classical arm needs no PCA).")
    p.add_argument("--results", default="results/regression_benchmark.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
