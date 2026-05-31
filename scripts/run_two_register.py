"""
Two-register quantum kernel experiments.

Loads antibody embeddings (igbert or balm_paired) and antigen embeddings
(esm2_650M) separately, concatenates them, and runs:
  - two_reg_minimal / two_reg_expressive  (core QuantAb claim)
  - quantum_minimal / quantum_expressive  (single-register ablation)
  - vqc                                   (variational regressor)
  - mlp / rbf_svm / random_forest         (classical baselines)

All methods receive the same concatenated [X_ab | X_tgt] features so the
only variable is the circuit / model architecture.

Usage
-----
  python scripts/run_two_register.py

  # Specific settings
  python scripts/run_two_register.py \\
      --ab-model igbert --tgt-model esm2_650M \\
      --n-components 6 --train-sizes 50 100 200 \\
      --no-vqc --overwrite

Output
------
  results/two_reg_<ab_model>_<tgt_model>_<n>d.json
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from sklearn.svm import SVR

from quantab.classical import MODELS as CLASSICAL_MODELS
from quantab.evaluation import learning_curve, spearman
from quantab.quantum_kernels import (
    build_minimal_kernel,
    build_expressive_kernel,
    build_two_register_minimal_kernel,
    build_two_register_expressive_kernel,
    VQCRegressor,
)

# Quantum is O(N²) — keep smaller; classical can go full dataset
QUANTUM_TRAIN_SIZES = [50, 100, 200, 500, 1000, 2000]
CLASSICAL_TRAIN_SIZES = [50, 100, 200, 500, 1000, 2000, 5000]

N_REPEATS = 5
TEST_SIZE = 500
RANDOM_STATE = 42


def load_embeddings(emb_dir: Path, model_key: str, n: int) -> np.ndarray:
    """Load the pre-reduced embedding array for a given model and dimension."""
    # Try unified first, then fall back to first found file
    candidate = emb_dir / model_key / f"unified_{n}d.npy"
    if candidate.exists():
        return np.load(candidate)
    # Try any *_{n}d.npy in the directory
    hits = sorted((emb_dir / model_key).glob(f"*_{n}d.npy"))
    if not hits:
        raise FileNotFoundError(
            f"No {n}d embedding found for '{model_key}' in {emb_dir / model_key}. "
            "Run extract_embeddings.py first."
        )
    if len(hits) > 1:
        print(f"  [warn] multiple {n}d files for {model_key}, using {hits[0].name}")
    return np.load(hits[0])


def load_labels(emb_dir: Path, model_key: str) -> np.ndarray:
    """Load normalised affinity labels (z-scored pKd)."""
    for name in ("unified_labels_norm.npy", "unified_labels.npy"):
        p = emb_dir / model_key / name
        if p.exists():
            return np.load(p)
    # Fall back to any *labels*.npy
    hits = sorted((emb_dir / model_key).glob("*labels*.npy"))
    if not hits:
        raise FileNotFoundError(f"No labels file found for '{model_key}' in {emb_dir / model_key}")
    return np.load(hits[0])


def run(args: argparse.Namespace) -> None:
    emb_dir = Path(args.embeddings_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    for n in args.n_components:
        tag = f"two_reg_{args.ab_model}_{args.tgt_model}_{n}d"
        out_path = results_dir / f"{tag}.json"

        if out_path.exists() and not args.overwrite:
            print(f"[{tag}] results exist — skipping. Use --overwrite to rerun.")
            continue

        print(f"\n[{tag}]  loading embeddings …")
        try:
            X_ab = load_embeddings(emb_dir, args.ab_model, n)
            X_tgt = load_embeddings(emb_dir, args.tgt_model, n)
            y = load_labels(emb_dir, args.ab_model)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        assert len(X_ab) == len(X_tgt) == len(y), (
            f"Length mismatch: X_ab={len(X_ab)}, X_tgt={len(X_tgt)}, y={len(y)}"
        )
        X_cat = np.hstack([X_ab, X_tgt])  # shape (N, 2*n)
        print(f"  N={len(X_cat)}  X_ab={X_ab.shape}  X_tgt={X_tgt.shape}  X_cat={X_cat.shape}")

        results: dict[str, dict] = {}
        q_sizes = [s for s in args.train_sizes if s <= args.quantum_max_train]
        c_sizes = args.train_sizes

        # --- Classical baselines ---
        for name, factory in CLASSICAL_MODELS.items():
            print(f"  {name} …")
            results[name] = learning_curve(
                factory(),
                X_cat, y,
                train_sizes=c_sizes,
                is_quantum=False,
                n_repeats=N_REPEATS,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

        # --- Classical on antibody-only (ablation: no antigen info) ---
        for name, factory in CLASSICAL_MODELS.items():
            key = f"{name}_ab_only"
            print(f"  {key} …")
            results[key] = learning_curve(
                factory(),
                X_ab, y,
                train_sizes=c_sizes,
                is_quantum=False,
                n_repeats=N_REPEATS,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

        if not args.no_quantum:
            # --- Single-register quantum kernels (ablation) ---
            for name, builder in [
                ("quantum_minimal", build_minimal_kernel),
                ("quantum_expressive", build_expressive_kernel),
            ]:
                print(f"  {name} ({2*n} qubits, concatenated input) …")
                kernel_fn = builder(n_qubits=2 * n)
                results[name] = learning_curve(
                    kernel_fn,
                    X_cat, y,
                    train_sizes=q_sizes,
                    is_quantum=True,
                    n_repeats=N_REPEATS,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                )

            # --- Two-register quantum kernels (core contribution) ---
            for name, builder in [
                ("two_reg_minimal", build_two_register_minimal_kernel),
                ("two_reg_expressive", build_two_register_expressive_kernel),
            ]:
                print(f"  {name} ({n}+{n} qubits, two-register) …")
                kernel_fn = builder(n_qubits_ab=n, n_qubits_tgt=n)
                results[name] = learning_curve(
                    kernel_fn,
                    X_cat, y,
                    train_sizes=q_sizes,
                    is_quantum=True,
                    n_repeats=N_REPEATS,
                    test_size=TEST_SIZE,
                    random_state=RANDOM_STATE,
                )

        # --- VQC regressor ---
        if not args.no_vqc:
            vqc_sizes = [s for s in args.train_sizes if s <= args.vqc_max_train]
            print(f"  vqc ({2*n} qubits) …")
            results["vqc"] = learning_curve(
                VQCRegressor(
                    n_qubits=2 * n,
                    n_layers=args.vqc_layers,
                    n_epochs=args.vqc_epochs,
                    lr=0.05,
                    random_state=RANDOM_STATE,
                ),
                X_cat, y,
                train_sizes=vqc_sizes,
                is_quantum=False,
                n_repeats=N_REPEATS,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {out_path}")

    print(f"\nDone. Results in {results_dir}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-register quantum kernel experiments.")
    p.add_argument("--embeddings-dir", default="embeddings")
    p.add_argument("--results-dir", default="results")
    p.add_argument(
        "--ab-model",
        default="igbert",
        choices=["igbert", "balm_paired"],
        help="Antibody embedding model.",
    )
    p.add_argument(
        "--tgt-model",
        default="esm2_650M",
        choices=["esm2_35M", "esm2_150M", "esm2_650M"],
        help="Antigen embedding model.",
    )
    p.add_argument(
        "--n-components",
        nargs="+",
        type=int,
        default=[6, 10],
        help="PCA dimensions per modality (default: 6 10).",
    )
    p.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=QUANTUM_TRAIN_SIZES,
        help="Training set sizes for the learning curve.",
    )
    p.add_argument(
        "--quantum-max-train",
        type=int,
        default=2000,
        help="Cap quantum kernel training sizes (O(N²) cost).",
    )
    p.add_argument(
        "--vqc-max-train",
        type=int,
        default=200,
        help="Cap VQC training sizes (slow per-sample forward pass).",
    )
    p.add_argument("--vqc-layers", type=int, default=3)
    p.add_argument("--vqc-epochs", type=int, default=100)
    p.add_argument("--no-quantum", action="store_true", help="Skip quantum kernels.")
    p.add_argument("--no-vqc", action="store_true", help="Skip VQC regressor.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
