# CLAUDE.md — Instructions for Claude Code

## Project context

QuantAb explores whether quantum kernel methods — specifically two-register quantum kernels in tensor product Hilbert spaces — can capture antibody-antigen binding affinity signal that classical methods miss, in the small-labeled-data regime.

This is a Briney Lab project at The Scripps Research Institute, originating from a hackathon with AWS compute support.

## Embedding architecture (two modalities, two registers)

Each sample produces **two independent embeddings** extracted by two separate models:

| Modality | Model | Input | Output |
|----------|-------|-------|--------|
| Antibody | IgBERT (preferred) or BALM | Paired VH+VL (concatenated with separator token) | Single pooled embedding [dim_ab] |
| Target protein | ESM2 | Target protein sequence | Pooled embedding [dim_target] |

**Paired VH+VL is required** — always use models that accept both chains together (IgBERT does this natively). Do not embed heavy and light chains separately unless a model forces it.

### Pipeline

```
Input CSV: heavy chain seq | light chain seq | target prot seq | affinity (KD)
                ↓                                     ↓
         IgBERT (paired VH+VL)                     ESM2
                ↓                                     ↓
       ab_embeddings [N, dim_ab]        target_embeddings [N, dim_target]
                ↓                                     ↓
         z-score normalize                    z-score normalize
                ↓                                     ↓
       PCA/PLSR → k_ab dims             PCA/PLSR → k_target dims
                ↓                                     ↓
        ┌───────────────────────────────────────────────┐
        │  Classical arm: concatenate → SVM / RF / MLP  │
        │  Quantum arm: two-register circuit             │
        │    register 1 (k_ab qubits): antibody         │
        │    register 2 (k_target qubits): target       │
        │    → entangling gates → kernel matrix → SVM   │
        │  Ablation: single-register on concatenated    │
        └───────────────────────────────────────────────┘
```

### Why two registers matter scientifically

Affinity is an interaction property — it emerges from the joint state of both molecules. The two-register quantum kernel operates in H_ab ⊗ H_target (tensor product), which is 2^(k_ab + k_target)-dimensional. Classical concatenation cannot represent this interaction structure without exponential cost. This is the core scientific claim.

### Dimensionality reduction

- **Normalize each modality independently** before any joint operation (different models, different scales).
- **Separate PCA per modality** for the hackathon (unsupervised, no label leakage).
- **PLSR** as a second DR arm for the paper (label-predictive, risk of overfitting in small-data regime — interesting tension to report).
- **Random projection** as a null baseline (one extra line of code, scientifically important).
- Number of components (k_ab, k_target) set by cross-validated Spearman ρ on a classical baseline, not fixed in advance.

### Qubit budget

Total qubits = k_ab + k_target. Typical targets: (6,6)=12, (8,8)=16, (10,10)=20, (14,14)=28.

**Repo:** https://github.com/brineylab/QuantAb — working directly on this repo (no fork). Each team member has their own branch: `benjamin` (bnemoz), others create their own. Always work on your branch, never commit directly to `master`.

## Current implementation status (single-register baseline)

The code currently on this branch implements the **single-register baseline / ablation arm** of the design above: antibody-only embeddings (no target register yet), reduced by PCA, fed to single-register quantum kernels and classical baselines. This is exactly the "single-register on concatenated features" ablation the two-register design is measured against — so it is a useful starting point, not a competing design. Building out the target modality and the two-register circuit is the next step.

What is implemented today:

- **Embedding models** — unified registry in `quantab/embeddings.py`:

  | Key | Model | Type | Dim | Input format | Pooling |
  |---|---|---|---|---|---|
  | `igbert` | `Exscientia/IgBert` | Antibody-specific | 1024 | Space-separated AAs | [CLS] |
  | `esm2_35M` | `facebook/esm2_t12_35M_UR50D` | General protein | 480 | Raw sequence | Mean |
  | `esm2_150M` | `facebook/esm2_t30_150M_UR50D` | General protein | 640 | Raw sequence | Mean |
  | `esm2_650M` | `facebook/esm2_t33_650M_UR50D` | General protein | 1280 | Raw sequence | Mean |
  | `balm_paired` | `brineylab/BALM-paired` | Natively-paired antibody | 1024 | (heavy, light) pair | [CLS] |

  Note: `balm_paired` takes heavy and light as two separate sequences (RoBERTa two-sequence input), not concatenated. (Caveat: `scripts/extract_embeddings.py` currently concatenates for all models — fix before running BALM through the script; the library `embed_and_reduce` path already handles pairing correctly.)

- **Quantum kernels** (`quantab/quantum_kernels.py`) — `quantum_minimal` (angle encoding + 1 entangling layer) and `quantum_expressive` (repeated Ry+CNOT layers), single-register, `lightning.qubit` simulator. O(N²) kernel matrices — keep ≤500 points.
- **Classical baselines** (`quantab/classical.py`) — linear/RBF/poly SVM + random forest.
- **Evaluation** (`quantab/evaluation.py`) — learning curves, k-fold CV, Spearman metric.
- **Scripts** — `extract_embeddings.py`, `run_experiments.py`, `generate_figures.py` (idempotent, CLI-driven).

## Dataset paths

DMS datasets are **not committed** (~162 MB). They are mirrored to S3 and also live on the hackathon server.

- **S3 (source of truth):** `s3://scrippsresearch-mshafiei-hackathon/DMS_Data/` and `.../AB-Bind-Database/` (readable by everyone in the hackathon AWS account `127696279288`). See `data/README.md` for download commands.
- **Server-local:** `/home/jovyan/work/Hackaton/DMS_Data/`

| Directory | Dataset | Key columns | N variants |
|---|---|---|---|
| `02_phillips_cr9114_cr6261/` | Phillips et al. 2021 CR9114 anti-influenza DMS | `heavy`, `light`, `*_neg_log_kd` | ~67,000 |
| `05_engelhart_aayl/` | Engelhart et al. AAYL variants | `heavy_chain_seq`, `binding_score`, `light_chain_seq` | ~varies |
| `06_shanehsazzadeh_trastuzumab/` | Shanehsazzadeh trastuzumab zero-shot binders | `HCDR3`, `-log(KD (M))` | 422 |
| `01_dailey_cr9114/` | Dailey CR9114 single-point mutants | `mut`, `minus_log_Kd` | ~varies |
| `03_magma_seq/` | MAGMA-seq data | mixed | — |
| `04_adams_4420/` | Adams et al. titeseq | see `16_titeseq/` | — |

```python
from pathlib import Path
from quantab.data import load_all, summarize

DATA_DIR = Path("/home/jovyan/work/Hackaton/DMS_Data")  # or /data/DMS_Data on EC2
df = load_all(DATA_DIR)
summarize(df)
```

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
├── data/                  # DMS datasets (not committed — .gitignore); data/README.md documents S3 location
├── embeddings/            # Extracted embeddings (not committed — .gitignore)
├── notebooks/             # Jupyter notebooks for exploration and visualization
│   └── 01_embedding_extraction.ipynb
│   └── 02_quantum_kernels.ipynb
│   └── 03_classical_baselines.ipynb
│   └── 04_learning_curves.ipynb
│   └── 05_results_visualization.ipynb
├── quantab/               # Python package
│   ├── __init__.py
│   ├── data.py            # DMS dataset loaders
│   ├── embeddings.py      # AbLM embedding extraction and PCA
│   ├── quantum_kernels.py # PennyLane quantum kernel implementations
│   ├── classical.py       # Classical kernel baselines
│   ├── evaluation.py      # Cross-validation, learning curves, metrics
│   └── visualization.py   # Plotting and figure generation
├── scripts/               # Standalone pipeline scripts
│   ├── extract_embeddings.py
│   ├── run_experiments.py
│   └── generate_figures.py
├── results/               # Experiment outputs (not committed — .gitignore)
├── figures/               # Generated figures
├── pyproject.toml
└── .gitignore
```

## Key dependencies

- `pennylane` — Quantum ML framework (two-register circuits, kernel estimation, simulation)
- `scikit-learn` — SVM, cross-validation, metrics, PCA, PLSR
- `polars` — Data manipulation
- `transformers` — HuggingFace model loading for IgBERT, BALM, ESM2
- `torch` — Backend for embedding extraction
- `matplotlib` / `seaborn` — Visualization
- `scipy` — Spearman correlation, statistical tests

## Scientific constraints

- **Primary metric:** Spearman rank correlation between predicted and measured binding affinity.
- **All comparisons must use identical reduced features.** Same PCA projection for quantum and classical arms. The only variable is the kernel/model.
- **Deterministic seeds everywhere.** Set `random_state` / `seed` in all stochastic operations. Default seed: 42.
- **Learning curves are the core analysis.** Every method must be evaluated across a range of training set sizes (e.g., 20, 50, 100, 200, full).
- **Never overfit to one dataset.** Hackathon uses 1 DMS dataset; paper will generalize to multiple. Keep code dataset-agnostic.
- **Never leak test data** — always split before fitting PCA.

## Tone and communication

This repo will be shared with labmates and PI. Code should be clean, well-documented, and presentation-ready. Notebooks should have clear markdown explanations between cells. Figures should be publication-quality from the start.

## What to prioritize

1. Get embedding extraction working first (this unblocks everything)
2. Get one quantum kernel + one classical baseline running end-to-end
3. Learning curve analysis
4. Then iterate: more kernels, more qubits, more models — and build out the target modality + two-register circuit

## AWS compute

**EC2 instance for simulation and training (launched 2026-05-29):**

| Field | Value |
|---|---|
| Instance ID | `i-038ef1f119518297a` |
| Type | `c5.4xlarge` (16 vCPUs, 32 GB RAM) |
| Private IP | `172.31.66.186` |
| Data volume | 100 GB EBS at `/dev/xvdf` → mounted at `/data` |
| AWS profile | `bnemoz-scripps` |
| Region | `us-west-2` |

Currently CPU-only (`lightning.qubit`). GPU quota increase pending — when granted, relaunch as `g4dn.xlarge` and switch to `lightning.gpu`.

**SSH via EICE** (key push window is 60 s — run all three commands in quick succession):

```bash
ssh-keygen -t ed25519 -N "" -f /tmp/eice-key -q <<< y

aws ec2-instance-connect send-ssh-public-key \
  --instance-id i-038ef1f119518297a \
  --instance-os-user ec2-user \
  --ssh-public-key file:///tmp/eice-key.pub \
  --profile bnemoz-scripps --region us-west-2

ssh -i /tmp/eice-key -o StrictHostKeyChecking=no \
  -o ProxyCommand='aws ec2-instance-connect open-tunnel \
    --instance-id i-038ef1f119518297a \
    --profile bnemoz-scripps --region us-west-2' \
  ec2-user@i-038ef1f119518297a
```

**Mount data volume (first login only):**

```bash
sudo mkfs.xfs /dev/xvdf && sudo mkdir /data
sudo mount /dev/xvdf /data && sudo chown ec2-user:ec2-user /data
```

**SCP a file to the instance:**

```bash
scp -i /tmp/eice-key \
  -o ProxyCommand='aws ec2-instance-connect open-tunnel --instance-id i-038ef1f119518297a --profile bnemoz-scripps --region us-west-2' \
  ./script.py ec2-user@i-038ef1f119518297a:/home/ec2-user/
```

## What to avoid

- Don't over-engineer infrastructure before we have results
- Don't chase quantum advantage claims — we're benchmarking, not advocating
- Don't use Qiskit unless there's a specific reason — PennyLane is the primary framework
