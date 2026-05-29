"""
Run learning curve experiments for all methods (quantum + classical) on pre-computed embeddings.

Requires: run extract_embeddings.py first.

Usage
-----
# All models, all methods, default settings
python scripts/run_experiments.py

# Specific model and dataset
python scripts/run_experiments.py --model igbert --dataset phillips_all --n-components 6

# Skip quantum (faster, classical only)
python scripts/run_experiments.py --no-quantum

Output
------
results/
  igbert_phillips_all_6d.json
  igbert_phillips_all_10d.json
  esm2_35M_phillips_all_6d.json
  ...

Each JSON contains learning curve data (train_sizes, means, stds) for every method.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from quantab.classical import MODELS as CLASSICAL_MODELS
from quantab.embeddings import REGISTRY
from quantab.evaluation import learning_curve
from quantab.quantum_kernels import build_expressive_kernel, build_minimal_kernel

# Training set sizes for learning curves
# Quantum is O(N²) — keep smaller. Classical can go larger.
QUANTUM_TRAIN_SIZES = [20, 50, 100, 200]
CLASSICAL_TRAIN_SIZES = [20, 50, 100, 200, 500, 1000]

N_REPEATS = 5
TEST_SIZE = 300
RANDOM_STATE = 42


def load_embeddings(embeddings_dir: Path, model_key: str, dataset: str, n_components: int):
    X = np.load(embeddings_dir / model_key / f"{dataset}_{n_components}d.npy")
    y = np.load(embeddings_dir / model_key / f"{dataset}_labels.npy")
    return X, y


def run(args: argparse.Namespace) -> None:
    emb_dir = Path(args.embeddings_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    models = list(REGISTRY.keys()) if args.model == "all" else [args.model]

    for model_key in models:
        model_dir = emb_dir / model_key
        if not model_dir.exists():
            print(f"No embeddings found for {model_key}, skipping. Run extract_embeddings.py first.")
            continue

        # Discover available datasets
        datasets = args.dataset if args.dataset else [
            p.stem.replace(f"_{args.n_components[0]}d", "")
            for p in sorted(model_dir.glob(f"*_{args.n_components[0]}d.npy"))
        ]

        for dataset in datasets:
            for n in args.n_components:
                tag = f"{model_key}_{dataset}_{n}d"
                out_path = results_dir / f"{tag}.json"

                if out_path.exists() and not args.overwrite:
                    print(f"[{tag}] results exist, skipping.")
                    continue

                try:
                    X, y = load_embeddings(emb_dir, model_key, dataset, n)
                except FileNotFoundError:
                    print(f"[{tag}] embeddings not found, skipping.")
                    continue

                print(f"\n[{tag}]  N={len(X)}  X.shape={X.shape}")
                results: dict[str, dict] = {}

                # --- Classical methods ---
                for name, factory in CLASSICAL_MODELS.items():
                    print(f"  {name}...")
                    results[name] = learning_curve(
                        factory(),
                        X, y,
                        train_sizes=CLASSICAL_TRAIN_SIZES,
                        is_quantum=False,
                        n_repeats=N_REPEATS,
                        test_size=TEST_SIZE,
                        random_state=RANDOM_STATE,
                    )

                # --- Quantum methods ---
                if not args.no_quantum:
                    for name, builder in [
                        ("quantum_minimal", build_minimal_kernel),
                        ("quantum_expressive", build_expressive_kernel),
                    ]:
                        print(f"  {name} (n_qubits={n})...")
                        kernel_fn = builder(n_qubits=n)
                        results[name] = learning_curve(
                            kernel_fn,
                            X, y,
                            train_sizes=QUANTUM_TRAIN_SIZES,
                            is_quantum=True,
                            n_repeats=N_REPEATS,
                            test_size=TEST_SIZE,
                            random_state=RANDOM_STATE,
                        )

                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Saved → {out_path}")

    print(f"\nDone. Results in {results_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learning curve experiments on pre-computed embeddings.")
    parser.add_argument(
        "--embeddings-dir",
        default="embeddings",
        help="Directory containing embedding files (default: embeddings/).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory to save result JSON files (default: results/).",
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "igbert", "esm2_35M", "esm2_150M"],
        help="Which model embeddings to use (default: all).",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        help="Dataset name(s) to run (default: all found in embeddings dir).",
    )
    parser.add_argument(
        "--n-components",
        nargs="+",
        type=int,
        default=[6, 10],
        help="PCA dimensions to run experiments on (default: 6 10).",
    )
    parser.add_argument(
        "--no-quantum",
        action="store_true",
        help="Skip quantum kernel methods (much faster, classical baselines only).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun and overwrite existing result files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
