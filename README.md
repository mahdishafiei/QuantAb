<p align="center">
  <img src="logo.png" alt="QuantAb logo" width="300">
</p>

# QuantAb

**Quantum kernel methods for antibody affinity prediction**

Can quantum kernels extract binding-relevant structure from antibody language model embeddings
that classical kernels miss — especially in the small-labeled-data regime typical of antibody
affinity prediction?

This project applies quantum machine learning to antibody-antigen affinity prediction. Antibody
sequences are encoded as embeddings via [IgBERT](https://huggingface.co/Exscientia/IgBert),
compressed to low dimensions via PCA (to match a qubit budget), and then evaluated using
quantum kernel SVMs against classical baselines across a range of training set sizes.

---

## Research question

> In the small-labeled-data regime characteristic of antibody affinity prediction, do quantum
> kernel methods operating on antibody language model embeddings capture binding-relevant
> structure that classical kernels miss?

---

## Pipeline

```
Antibody sequences
      ↓
IgBERT embeddings  (1024-dim, [CLS] token)
      ↓
PCA reduction  (6 or 10 dimensions — matching qubit budget)
      ↓
┌─────────────────────────┐    ┌──────────────────────────────┐
│  Quantum kernel SVM     │    │  Classical kernel SVM / RF   │
│  (PennyLane circuits)   │    │  (RBF, Linear, Polynomial)   │
└─────────────────────────┘    └──────────────────────────────┘
      ↓                                      ↓
      └──────── Spearman ρ vs. training set size (learning curves) ──────┘
```

---

## Methods

| Component | Choice | Notes |
|---|---|---|
| Embedding model | IgBERT (`Exscientia/IgBert`) | 1024-dim BERT-based antibody LM |
| Dimensionality reduction | PCA | 6 and 10 components (= n_qubits) |
| Quantum framework | PennyLane `lightning.qubit` | Statevector simulation |
| Quantum kernels | Minimal ansatz, Expressive ansatz | Angle encoding + CNOT layers |
| Classical baselines | Linear SVM, RBF SVM, Poly SVM, Random Forest | Identical PCA features |
| Primary metric | Spearman rank correlation | Robust to affinity scale differences |
| Core analysis | Learning curves | Performance vs. training set size |

---

## Datasets

All experiments use publicly available deep mutational scanning (DMS) datasets with
continuous binding measurements (KD or enrichment score):

| Dataset | Antibody | Variants | Measurement |
|---|---|---|---|
| Phillips et al. 2021 | CR9114 anti-influenza | ~67,000 | -log(KD) |
| Engelhart et al. | AAYL antibody variants | ~42,000 | Binding score |
| Shanehsazzadeh et al. | Trastuzumab (HER2) | 422 | -log(KD) |

---

## Repository structure

```
QuantAb/
├── quantab/               # Core Python package
│   ├── data.py            # DMS dataset loaders
│   ├── embeddings.py      # IgBERT extraction + PCA
│   ├── quantum_kernels.py # PennyLane quantum kernel implementations
│   ├── classical.py       # Classical kernel baselines
│   ├── evaluation.py      # Learning curves, CV, Spearman metric
│   └── visualization.py   # Figure generation
├── scripts/               # End-to-end pipeline scripts
├── notebooks/             # Exploration and results notebooks
├── figures/               # Generated figures (committed)
├── PLAN.md                # Full experimental design
├── CLAUDE.md              # Claude Code collaboration guide
└── MEMORY.md              # Design decisions and discussion log
```

---

## Getting started

```bash
git clone git@github.com:mahdishafiei/QuantAb.git
cd QuantAb
pip install -e .
```

```python
from pathlib import Path
from quantab.data import load_all, summarize
from quantab.embeddings import load_igbert, embed_and_reduce

DATA_DIR = Path("/path/to/DMS_Data")
df = load_all(DATA_DIR)
summarize(df)

tok, model = load_igbert()
X, y, pca = embed_and_reduce(df, tok, model, n_components=10,
                              cache_path=Path("embeddings/all_10d.npy"))
```

---

## Team

Briney Lab, Department of Immunology and Microbiology, The Scripps Research Institute

---

## License

TBD
