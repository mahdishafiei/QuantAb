"""
Generate publication-quality figures from experiment results.

Usage
-----
python scripts/generate_figures.py
python scripts/generate_figures.py --results-dir results --figures-dir figures
"""

import argparse
import json
from pathlib import Path

from quantab.visualization import plot_learning_curves


def run(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        print(f"No result files found in {results_dir}/. Run run_experiments.py first.")
        return

    for result_file in result_files:
        with open(result_file) as f:
            results = json.load(f)

        tag = result_file.stem
        fig_path = figures_dir / f"learning_curve_{tag}.png"

        # Parse tag for a readable title: igbert_phillips_all_6d → IgBERT | phillips_all | 6 qubits
        parts = tag.split("_")
        model = parts[0].upper() if parts[0] == "igbert" else parts[0].replace("esm2", "ESM-2")
        dims = next((p for p in parts if p.endswith("d") and p[:-1].isdigit()), "?")
        dataset = "_".join(p for p in parts[1:] if not p.endswith("d") or not p[:-1].isdigit())
        title = f"{model}  ·  {dataset}  ·  {dims[:-1]} components"

        fig = plot_learning_curves(results, title=title, save_path=fig_path)
        print(f"Saved → {fig_path}")

    print(f"\nAll figures in {figures_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures from experiment results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
