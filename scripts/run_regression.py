"""Regression benchmark on the model embeddings — leak-free.

Predicts normalized binding affinity (``affinity_norm``) from each model's mean-pooled
embeddings and reports Spearman ρ. Guards against two kinds of leakage:

- PCA / scaling leakage: any PCA / scaling lives inside an sklearn Pipeline, so it is
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
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ANTIBODY_MODELS = ["igbert", "esm2_650M", "balm_paired"]
METRICS = ["spearman", "pearson", "r2", "rmse", "recall@10%"]


def compute_metrics(y_true, y_pred) -> dict:
    """Regression + retrieval metrics for comparing models on affinity prediction."""
    n = len(y_true)
    k = max(1, int(round(0.10 * n)))                       # top 10%
    top_true = set(np.argsort(y_true)[-k:])
    top_pred = set(np.argsort(y_pred)[-k:])
    return {
        "spearman": float(spearmanr(y_true, y_pred).statistic),   # rank correlation
        "pearson": float(pearsonr(y_true, y_pred)[0]),            # linear correlation
        "r2": float(r2_score(y_true, y_pred)),                    # variance explained
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),  # error (std units)
        "recall@10%": len(top_true & top_pred) / k,               # strong-binder retrieval
    }


def build_estimator(pca_dims: int) -> Pipeline:
    """Lab-style MLP head; optional in-fold PCA (fit per training fold → no leak)."""
    steps = [("scaler", StandardScaler())]
    if pca_dims and pca_dims > 0:
        steps.append(("pca", PCA(n_components=pca_dims, random_state=42)))
    steps.append(("mlp", MLPRegressor(
        hidden_layer_sizes=(64, 32), early_stopping=True,
        validation_fraction=0.1, max_iter=200, random_state=42)))
    return Pipeline(steps)


def cv_metrics(X, y, groups, pca_dims, n_splits=5) -> dict:
    """Cross-validated metrics. Returns {metric: {mean, std}} averaged over folds."""
    if groups is not None:
        n_splits = min(n_splits, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_splits).split(X, y, groups)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X)
    per_fold = []
    for tr, te in splitter:
        est = build_estimator(pca_dims)
        est.fit(X[tr], y[tr])
        per_fold.append(compute_metrics(y[te], est.predict(X[te])))
    return {
        mtr: {"mean": float(np.mean([f[mtr] for f in per_fold])),
              "std": float(np.std([f[mtr] for f in per_fold]))}
        for mtr in METRICS
    }


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
    hdr = f"{'model':13s} " + " ".join(f"{m:>11s}" for m in METRICS)
    print("\n" + hdr); print("-" * len(hdr))
    for m in models:
        X = np.load(emb / m / "unified_raw.npy")
        y = np.load(emb / m / "unified_labels_norm.npy")
        ab = cv_metrics(X, y, groups, args.pca)
        results[m] = {"antibody": ab}
        print(f"{m:13s} " + " ".join(f"{ab[mt]['mean']:>11.3f}" for mt in METRICS))
        if antigen is not None:
            results[m]["antibody_antigen"] = cv_metrics(np.hstack([X, antigen]), y, groups, args.pca)

    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"group_col": args.group_col, "pca": args.pca, "antigen": args.antigen,
         "metrics": METRICS, "results": results},
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
