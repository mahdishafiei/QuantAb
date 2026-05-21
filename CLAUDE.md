# CLAUDE.md — Instructions for Claude Code

## Project context

QuantAb explores whether quantum kernel methods can extract binding-relevant signal from antibody language model (AbLM) embeddings that classical kernels miss, particularly in the small-labeled-data regime typical of affinity prediction.

This is a Briney Lab project at The Scripps Research Institute, originating from a hackathon with AWS compute support.

## Codebase conventions

- **Python only.** All analysis code in Python 3.11+.
- **Polars, not pandas.** Use `polars` for all tabular data manipulation. Never use pandas.
- **Pathlib always.** Use `from pathlib import Path` for all file handling. No `os.path`.
- **Progress bars.** Use `from tqdm.auto import tqdm` for any loop or batch operation.
- **Idempotent scripts.** Every pipeline script should be safe to re-run. Check for existing outputs, use deterministic seeds, skip completed steps.
- **Type hints.** Use type hints on all function signatures.
- **Docstrings.** Google-style docstrings on all public functions and classes.

## Project structure

```
QuantAb/
├── README.md
├── PLAN.md
├── CLAUDE.md
├── MEMORY.md
├── data/                  # DMS datasets (not committed — .gitignore)
├── embeddings/            # Extracted embeddings (not committed — .gitignore)
├── notebooks/             # Jupyter notebooks for exploration and visualization
│   └── 01_embedding_extraction.ipynb
│   └── 02_quantum_kernels.ipynb
│   └── 03_classical_baselines.ipynb
│   └── 04_learning_curves.ipynb
│   └── 05_results_visualization.ipynb
├── quantab/               # Python package
│   ├── __init__.py
│   ├── embeddings.py      # AbLM embedding extraction and PCA
│   ├── quantum_kernels.py # PennyLane quantum kernel implementations
│   ├── classical.py       # Classical kernel baselines
│   ├── evaluation.py      # Cross-validation, learning curves, metrics
│   └── visualization.py   # Plotting and figure generation
├── scripts/               # Standalone pipeline scripts
│   ├── extract_embeddings.py
│   ├── run_quantum_experiment.py
│   ├── run_classical_baselines.py
│   └── generate_figures.py
├── results/               # Experiment outputs (not committed — .gitignore)
├── figures/               # Generated figures
├── pyproject.toml
└── .gitignore
```

## Key dependencies

- `pennylane` — Quantum ML framework (circuits, kernels, simulation)
- `scikit-learn` — SVM, cross-validation, metrics
- `polars` — Data manipulation
- `transformers` — HuggingFace model loading for AbLM checkpoints
- `torch` — Backend for embedding extraction
- `matplotlib` / `seaborn` — Visualization
- `scipy` — Spearman correlation, statistical tests

## Scientific constraints

- **Primary metric:** Spearman rank correlation between predicted and measured binding affinity.
- **All comparisons must use identical reduced features.** Same PCA projection for quantum and classical arms. The only variable is the kernel/model.
- **Deterministic seeds everywhere.** Set `random_state` / `seed` in all stochastic operations. Default seed: 42.
- **Learning curves are the core analysis.** Every method must be evaluated across a range of training set sizes (e.g., 20, 50, 100, 200, full).
- **Never overfit to one dataset.** Hackathon uses 1 DMS dataset; paper will generalize to multiple. Keep code dataset-agnostic.

## Tone and communication

This repo will be shared with labmates and PI. Code should be clean, well-documented, and presentation-ready. Notebooks should have clear markdown explanations between cells. Figures should be publication-quality from the start.

## What to prioritize

1. Get embedding extraction working first (this unblocks everything)
2. Get one quantum kernel + one classical baseline running end-to-end
3. Learning curve analysis
4. Then iterate: more kernels, more qubits, more models

## What to avoid

- Don't over-engineer infrastructure before we have results
- Don't chase quantum advantage claims — we're benchmarking, not advocating
- Don't use Qiskit unless there's a specific reason — PennyLane is the primary framework
