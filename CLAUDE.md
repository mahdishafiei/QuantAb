# CLAUDE.md — QuantAb Project Guide

## What this project does

QuantAb tests whether quantum kernel methods outperform classical kernels (SVM, random forest)
when predicting antibody binding affinity from language model embeddings — especially in the
small-labeled-data regime that is typical of real antibody engineering campaigns.

Pipeline: antibody sequences → IgBERT embeddings (1024-dim) → PCA (6 or 10 dims) →
quantum kernel SVM or classical kernel SVM → Spearman correlation with measured KD.

The central analysis is **learning curves**: how does each method's Spearman ρ change as
training set size grows from 20 to 500+ examples?

---

## Environment

Everything runs in the base conda environment on this server. All dependencies are installed.

```bash
# Verify
python -c "import pennylane, polars, sklearn, transformers, torch; print('ok')"
```

Key package versions: PennyLane 0.45, polars 1.39, torch 2.11+cu130, transformers 5.5.

---

## Repository layout

```
QuantAb/
├── quantab/
│   ├── data.py            # Dataset loaders → (heavy, light, affinity, dataset) DataFrames
│   ├── embeddings.py      # IgBERT extraction + PCA
│   ├── quantum_kernels.py # PennyLane quantum kernels (minimal + expressive)
│   ├── classical.py       # Sklearn classical baselines
│   ├── evaluation.py      # Learning curves, k-fold CV, Spearman metric
│   └── visualization.py   # Matplotlib figure generation
├── scripts/
│   ├── extract_embeddings.py     # Run IgBERT on all datasets → save to embeddings/
│   ├── run_quantum_experiment.py # Load embeddings → quantum kernel learning curves
│   ├── run_classical_baselines.py
│   └── generate_figures.py
├── notebooks/
│   ├── 01_embedding_extraction.ipynb
│   ├── 02_quantum_kernels.ipynb
│   ├── 03_classical_baselines.ipynb
│   ├── 04_learning_curves.ipynb
│   └── 05_results_visualization.ipynb
├── data/        # DMS datasets — NOT committed (see paths below)
├── embeddings/  # Cached IgBERT embeddings — NOT committed
├── results/     # Experiment outputs — NOT committed
└── figures/     # Generated figures — committed
```

---

## Dataset paths (server-local, not in repo)

All DMS datasets live at: `/home/jovyan/work/Hackaton/DMS_Data/`

| Directory | Dataset | Format | Key columns | N variants |
|---|---|---|---|---|
| `02_phillips_cr9114_cr6261/` | Phillips et al. 2021 CR9114 anti-influenza DMS | CSV | `heavy`, `light`, `*_neg_log_kd` | ~67,000 |
| `05_engelhart_aayl/` | Engelhart et al. AAYL antibody variants | CSV | `heavy_chain_seq`, `binding_score`, `light_chain_seq` | ~varies |
| `06_shanehsazzadeh_trastuzumab/` | Shanehsazzadeh trastuzumab zero-shot binders | CSV | `HCDR3`, `-log(KD (M))` | 422 |
| `01_dailey_cr9114/` | Dailey CR9114 single-point mutants | CSV | `mut`, `minus_log_Kd` | ~varies |
| `03_magma_seq/` | MAGMA-seq data | Mixed | See SD files | — |
| `04_adams_4420/` | Adams et al. titeseq | CSV | See 16_titeseq/ subdir | — |

Load all datasets in one call:

```python
from pathlib import Path
from quantab.data import load_all, summarize

DATA_DIR = Path("/home/jovyan/work/Hackaton/DMS_Data")
df = load_all(DATA_DIR)
summarize(df)
```

---

## Embedding models

Two models are supported via a unified interface in `quantab/embeddings.py`:

| Key | Model | Type | Dim | Input format | Pooling |
|---|---|---|---|---|---|
| `igbert` | `Exscientia/IgBert` | Antibody-specific | 1024 | Space-separated AAs | [CLS] |
| `esm2_35M` | `facebook/esm2_t12_35M_UR50D` | General protein | 480 | Raw sequence | Mean |
| `esm2_150M` | `facebook/esm2_t30_150M_UR50D` | General protein | 640 | Raw sequence | Mean |

The IgBERT vs ESM-2 comparison is scientifically central: does antibody-specific pretraining
give quantum kernels better structure to exploit?

```python
from quantab.embeddings import load_model, embed_and_reduce, REGISTRY
from pathlib import Path

# Load any model by key
tok, model, cfg = load_model("igbert")       # or "esm2_35M", "esm2_150M"

X, y, pca = embed_and_reduce(
    df,
    tok,
    model,
    cfg,
    n_components=10,           # PCA dims = n_qubits
    cache_path=Path("embeddings/phillips_igbert_10d.npy"),
)
```

Always use `cache_path` — IgBERT on 67k sequences takes ~30 min on CPU.
Name cache files as `{dataset}_{model_key}_{n_components}d.npy` for clarity.

---

## Quantum kernels

Two architectures, both in `quantab/quantum_kernels.py`:

| Name | Circuit | Use |
|---|---|---|
| `quantum_minimal` | Angle encoding + 1 CNOT layer | Baseline quantum |
| `quantum_expressive` | Angle encoding + 2 repeated Ry+CNOT layers | More expressive |

```python
from quantab.quantum_kernels import build_minimal_kernel, build_expressive_kernel

k = build_minimal_kernel(n_qubits=6)   # n_qubits must match PCA n_components
k(x1, x2)  # returns scalar in [0, 1]
```

Kernel matrix computation is O(N²) — use subsampled datasets (≤500 points) for experiments.
Simulator: `lightning.qubit` (fast CPU statevector).

---

## Classical baselines

```python
from quantab.classical import MODELS

# Available: "linear_svm", "rbf_svm", "poly_svm", "random_forest"
model = MODELS["rbf_svm"]()
model.fit(X_train, y_train)
```

---

## Evaluation

Primary metric: **Spearman rank correlation** between predicted and measured affinity.

```python
from quantab.evaluation import learning_curve, evaluate_classical

# Classical learning curve
results = learning_curve(
    MODELS["rbf_svm"](),
    X, y,
    train_sizes=[20, 50, 100, 200, 500],
    is_quantum=False,
)

# Quantum learning curve
from quantab.quantum_kernels import build_minimal_kernel
results = learning_curve(
    build_minimal_kernel(n_qubits=6),
    X, y,
    train_sizes=[20, 50, 100, 200],
    is_quantum=True,
)
```

---

## Hackathon deliverables (MVP)

1. Embedding extraction on Phillips + Engelhart datasets → cached `.npy` files
2. Learning curves for: `quantum_minimal`, `quantum_expressive`, `linear_svm`, `rbf_svm`, `random_forest`
3. Two PCA dimensions: 6 and 10 qubits
4. Summary figure: Spearman ρ vs training set size, all methods on one plot
5. Scatter plot: best method predicted vs measured affinity

---

## Scientific constraints

- **Same PCA projection for all methods** — only the kernel/model varies, nothing else.
- **Deterministic seeds everywhere.** Default seed: 42.
- **Never leak test data** — always split before fitting PCA.
- **Primary metric is Spearman ρ**, not RMSE or R².

---

## Conventions

- **Polars, not pandas** for all tabular data.
- **Pathlib** for all file paths.
- **tqdm.auto** for any loop over sequences or batches.
- **Type hints** on all function signatures.
- **Google-style docstrings** on all public functions.
- Scripts must be idempotent (check for existing outputs, skip if present).
