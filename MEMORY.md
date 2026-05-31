# MEMORY.md — Open Questions and Discussion Log

## Open questions

### Dataset selection (HIGH PRIORITY)

Which DMS dataset to use for the hackathon? Requirements:
- >200 variants with measured binding (KD or enrichment score)
- Continuous affinity measurement (not binary binder/non-binder)
- Publicly available sequences
- Ideally against a well-characterized antigen

**Candidates to evaluate:**
- Taft et al. — therapeutic antibody DMS
- Adams et al. — antibody DMS with binding
- Phillips et al. (2021) — CR9114 anti-influenza
- Mason et al. — SARS-CoV-2 RBD DMS datasets
- Starr et al. — SARS-CoV-2 RBD deep scanning

**Status:** Loaders are now implemented in `quantab/data.py` for Phillips CR9114 (~67k, paired, continuous -log KD — strongest candidate), Engelhart AAYL (binding score), and Shanehsazzadeh trastuzumab (422, HCDR3 + KD). Dailey/MAGMA-seq/Adams deferred (single-point or complex multi-file formats). All datasets are mirrored to `s3://scrippsresearch-mshafiei-hackathon/DMS_Data/`.

**Action:** Confirm which one is primary for the hackathon run. Phillips is the cleanest large paired set.

---

### Which AbLM checkpoints?

The scaling laws paper produced 15 models (5 sizes × 3 data sizes). For the hackathon we need 3.

**Proposed selection:**
- Smallest model / largest data
- Medium model / largest data
- Largest model / largest data

Rationale: fix data size, vary model capacity — cleanest comparison axis. But should we instead fix model size and vary data? Or pick along the Pareto frontier?

**Interim:** the implemented registry uses public checkpoints (IgBERT, ESM-2 35M/150M/650M, BALM-paired) so the pipeline runs end-to-end today; the custom scaling-laws checkpoints are deferred to the full paper where the scale comparison can be done systematically.

**Action:** Discuss with Mahdi — he knows the models best.

---

### Embedding layer selection

Which layer(s) of the AbLM to extract embeddings from?
- Last hidden state (most common)?
- Pool across layers?
- CDR-only tokens vs. full sequence?

CDR-only might be more relevant for binding but reduces token count. Need to decide before extraction. (Current code uses last hidden state — [CLS] pooling for IgBERT/BALM, mean pooling for ESM-2.)

---

### PCA vs. other dimensionality reduction

PCA is the default and most interpretable. But:
- UMAP could preserve local structure better (relevant for kernel methods)
- Learned linear projections (supervised dimensionality reduction) could be stronger but risk overfitting with small data
- Random projections as a null baseline?

**Decision for hackathon:** PCA. Keep it simple. Explore alternatives in the paper.

---

### Quantum circuit architecture

Which ansatz designs to test? PennyLane options include:
- `AngleEmbedding` + `StronglyEntanglingLayers` (expressive, deep)
- `AngleEmbedding` + `BasicEntanglerLayers` (simpler, shallower)
- Custom hardware-efficient ansatz

How many layers/depth? More depth = more expressiveness but slower simulation and potential barren plateaus.

**Decision for hackathon:** Start with 2 architectures (one simple, one expressive). Systematic depth sweep in the paper. A single-register implementation of both (`quantum_minimal`, `quantum_expressive`) already exists in `quantab/quantum_kernels.py` and serves as the ablation baseline; the two-register version is the next build.

---

### Noise simulation

Should we simulate quantum noise (shot noise, gate errors, decoherence) or use ideal statevector simulation?

- Ideal simulation is cleaner for the scientific question (does the Hilbert space structure help?)
- Noisy simulation is more realistic for near-term hardware claims

**Decision for hackathon:** Ideal simulation. Add noise models in the paper as a separate analysis.

---

### AWS compute allocation

**Instance launched 2026-05-29:** `i-038ef1f119518297a`, `c5.4xlarge`, private IP `172.31.66.186`, 100 GB EBS at `/data`. See CLAUDE.md for SSH instructions.

GPU quota for G/VT instances is currently 0 on the hackathon account — organizers need to request an increase. When approved, relaunch as `g4dn.xlarge` (T4 GPU) and switch PennyLane device to `lightning.gpu` for faster kernel matrix computation.

For Amazon Braket (real quantum hardware): possible future extension, not in hackathon scope.

---

## ⚠ CRITICAL ISSUES — must be enforced in every iteration, especially the final one

> **For Mahdi and Terrence:** these are non-negotiable correctness requirements, not preferences.
> They also need to be acknowledged explicitly in the paper Methods and/or Discussion.

---

### ISSUE 1 — Data leakage via PCA fitted on the full dataset

**What:** PCA (and any other unsupervised DR step) must be fitted **on the training split only**,
then applied (transform-only) to the validation and test splits.

**Why it matters:** Fitting PCA on all rows — including the test set — leaks test-set distributional
information into the feature space. Results will be optimistically biased. This is particularly
damaging for learning-curve analyses where small training sets are compared to large ones:
the PCA "saw" all 40k sequences regardless of training size.

**Enforcement:**
```python
# CORRECT
pca.fit(X_train)
X_train_r = pca.transform(X_train)
X_test_r  = pca.transform(X_test)   # never fit on test

# WRONG — what the current pipeline likely does
pca.fit(X_all)   # leaks test distribution into PCA axes
```

The current `unified_6d/10d.npy` embeddings on S3 were computed on all 40k rows — they
**cannot be used as-is for honest evaluation**. PCA must be re-run per train/test fold.
Raw embeddings (`unified_raw.npy`, ~160–200 MB per model) are on S3 and can be used
to redo this correctly without re-running model inference.

**Paper mention:** Methods — note that PCA projection is fitted on the training split of each
cross-validation fold and applied to the held-out fold.

---

### ISSUE 2 — CLS token pooling is not acceptable

**What:** Do **not** use the CLS token as the sequence embedding for IgBERT or BALM-paired.
Use **mean pooling over all position embeddings** instead.

**Why it matters:** The CLS token was trained for sequence-level classification tasks. There is
no guarantee it encodes binding-interface geometry or CDR-loop conformation. For affinity
prediction, the signal lives in specific positions (especially HCDR3); CLS may ignore them
entirely. Mean pooling is lossy too, but at least every position — including HCDR3 — contributes.

**Enforcement:** Mean pooling over the full recombined VH+VL sequence (post-VDJ — HCDR3 is
present). Do not truncate to V-gene only (that cuts off before the CDR3 junction).

```python
# CORRECT — mean over all token positions (excluding special tokens)
embedding = model_output.last_hidden_state[:, 1:-1, :].mean(dim=1)

# WRONG — current Mahdi implementation
embedding = model_output.last_hidden_state[:, 0, :]  # CLS token only
```

The current `unified_raw.npy` / `unified_6d/10d.npy` files were computed with CLS pooling
for IgBERT and BALM-paired — **they need to be regenerated** with mean pooling.
ESM2 already uses mean pooling and is unaffected.

**Paper mention:** Methods — state mean pooling was used. Discussion — acknowledge this as a
known limitation; ideal solution is a length-constrained model that preserves per-position
resolution without requiring pooling.

---

## Design decisions (settled)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Quantum framework | PennyLane | Python-native, strong ML integration, good tutorials, quantum kernel support built-in |
| Primary metric | Spearman ρ | Rank correlation is robust to scale; standard in binding prediction benchmarks |
| Prediction target | Regression on log₁₀(KD/M) per dataset | = `-(log10(kd_nm) - 9)`. Continuous regression, not binary. Evaluated with Spearman ρ + RMSE. Never predict absolute KD across datasets (assay dynamic ranges are incomparable). |
| Core analysis | Learning curves | Directly tests the small-data hypothesis; publishable regardless of outcome |
| Hackathon qubit counts | (6,6) and (10,10) per register | 12 and 20 total qubits. (8,8)=16 dropped (not a useful scaling point). 28 qubits (14+14) reserved for paper, GPU only. |
| Antibody model | IgBERT (preferred) or BALM | Must accept **paired VH+VL** — never embed chains separately |
| Target model | ESM2 | Standard protein LM; pooled sequence embedding |
| Pooling strategy | Mean pooling over full sequence | CLS rejected (no guarantee it encodes binding-relevant features). Mean over full recombined VH+VL (post-VDJ — HCDR3 is included). V-gene alone is NOT enough (cuts off before HCDR3 junction). Diluted but present. Long-term fix: length-constrained model for full per-position embeddings. |
| Embedding pipeline | Separate extraction → separate normalize → separate PCA → two-register quantum / concatenate classical | Modalities are heterogeneous; joint PCA without normalization is dominated by largest space |
| DR method (hackathon) | PCA per modality | Unsupervised, no label leakage, interpretable |
| DR method (paper) | PCA + PLSR + random projection | PLSR maximizes label covariance; random projection is null baseline |
| Quantum circuit topology | Two-register (ab register ‖ target register) + entangling gates | Tensor product H_ab ⊗ H_target naturally represents interaction; exponentially richer than concatenation |
| Key ablation | Single-register on concatenated features | Isolates benefit of tensor product structure from quantum kernel benefit generally |
| Classical baselines | Linear/RBF/Polynomial SVM + Random Forest + MLP | Standard kernel zoo, non-kernel reference, and neural baseline |
| Data storage | S3 mirror, not committed to git | DMS data ~162 MB; `s3://scrippsresearch-mshafiei-hackathon/`, see `data/README.md` |
| Approach to results | Benchmark, not advocacy | We report what we find. Negative results are valuable. |

---

## IgBERT implementation notes

- Model ID: `Exscientia/IgBert`; hidden dim 1024 (BERT-large scale).
- Input format: space-separated amino acids — `"E V Q L V E S ..."`.
- Sequence embedding = **mean pooling over residue positions** (excluding special tokens), per ISSUE 2 — not the CLS token.
- On CPU: ~30 min for 67k sequences; always pass `cache_path` to `embed_and_reduce()` to save raw embeddings. Name caches `{dataset}_{model_key}_{n_components}d.npy`.
- Load report shows UNEXPECTED `cls.predictions` keys — expected and harmless (MLM head unused).

---

## Discussion log

### 2026-05-29 — Dataset curation and architecture revision (Benjamin + Claude)

**DMS data curated.** Five datasets processed from `DMS_Data/` into standardised schema
(`hc_seq, lc_seq, target_seq, kd_nm, source, antibody, antigen, notes`). Script: `scripts/curate_dms_data.py`.
After filtering incomplete rows (no target, no LC, no KD): **71,239 rows** in `curated_data/filtered_data.csv`.
Data lives on S3 — both local data dirs are gitignored:
- Raw DMS: `s3://scrippsresearch-mshafiei-hackathon/DMS_Data/`
- Curated: `s3://scrippsresearch-mshafiei-hackathon/curated_data/filtered_data.csv`

| Source | Rows | Notes |
|--------|------|-------|
| Phillips 2021 | 66,112 | CR9114/CR6261 vs H1/H3 HA — full VH+VL |
| Dailey | 4,974 | CR9114 single-mutants vs H1/H3 HA — VH/VL from PDB |
| MAGMA-seq | 153 | 4A8/CC121 vs SARS-CoV-2 RBD — VH/VL from PDB |

**⚠ DATA IMBALANCE — action required before embedding extraction.**
Phillips is 93% of the data (one antibody lineage against one antigen family). Pooling without
balancing makes all metrics Phillips-dominated and scientifically misleading. Data input will be
modified. Options: stratified cap per source (~5k rows each) OR cross-distribution split
(train on Phillips, test generalization on Dailey + MAGMA-seq). The cross-dataset framing
is the stronger scientific story and naturally tests the small-data hypothesis.

**Architecture revised for actual dataset size (71k rows):**
- Qubit plan: hackathon uses **(6,6)=12 and (10,10)=20** only. (8,8)=16 dropped (not a useful
  scaling point). (14,14)=28 reserved for paper, GPU required.
- Quantum kernel ceiling: N≈2000 training samples on CPU (20 qubits ≈ 2h). Stay ≤2000 for
  quantum; classical baselines can use full N.
- PCA dims: no longer data-limited at 71k rows. Set by explained variance elbow on actual
  embeddings (expect ~30–60 PCs for IgBERT). Do not fix in advance.
- Learning curves: N = 50, 100, 200, 500, 1000, 2000.
- "Small-data" framing recovers via cross-distribution generalization, not random subsampling.

---

### 2026-05-31 — Unified dataset + embeddings ready (Terrence + Mahdi + Benjamin)

**Everything needed for quantum kernel experiments is on S3 and downloaded locally.**

**Terrence contributed:** `ANDD_dataset.xlsx` — merged Ab/Nb–antigen affinities from SAbDab,
sdAb-DB, abYbank, PDB, PLAbDab-nano, INDI, and other literature/SPR sources. 3,310 usable rows
after filtering (heavy chain + antigen sequence + sane measured KD 1pM–1mM, no predicted values).

**Unified dataset** (`curated_data/unified_affinity.csv`): **40,937 rows, 1,041 antigens**.

| Dataset | Rows | Note |
|---------|------|------|
| Phillips 2021 | 33,345 | H3-HA arm dropped (88% at ceiling) |
| Dailey | 4,196 | CR9114 single-mutants |
| ANDD | 3,310 | literature/SPR, diverse antigens |
| MAGMA-seq | 86 | SARS-CoV-2 |

Regression target: `affinity_norm` (pKd z-scored within `norm_group`). Raw `affinity` (pKd 3.0–12.0) kept for reference.

**Embeddings computed and synced to `embeddings/` (local) and S3:**

| Model | Role | Files | Shape |
|-------|------|-------|-------|
| IgBERT | Antibody | `unified_6d.npy`, `unified_10d.npy` | (40937, 6/10) |
| BALM-paired | Antibody | `unified_6d.npy`, `unified_10d.npy` | (40937, 6/10) |
| ESM2-650M | Antibody + **Antigen** | `unified_6d/10d.npy` + `unified_antigen_6d/10d.npy` | (40937, 6/10) |

**Two-register quantum kernel is unblocked.** Register 1 (k_ab qubits): IgBERT or BALM-paired
`unified_6d/10d.npy`. Register 2 (k_tgt qubits): ESM2-650M `unified_antigen_6d/10d.npy`.
Labels: `unified_labels_norm.npy` (z-scored, use this as target).

Teammate sync:
```bash
aws s3 sync s3://scrippsresearch-mshafiei-hackathon/embeddings/ embeddings/ \
  --profile <your-profile> --region us-west-2 --exclude "*_raw.npy" --exclude "curated_*"
aws s3 cp s3://scrippsresearch-mshafiei-hackathon/curated_data/unified_affinity.csv \
  curated_data/ --profile <your-profile> --region us-west-2
```

---

### 2026-05-29 — Single-register pipeline + dataset mirror merged (Mahdi)

- Implemented the full single-register baseline pipeline on the `mahdi_quant` branch: `quantab/data.py` (Phillips/Engelhart/trastuzumab loaders), `embeddings.py` (IgBERT + ESM-2 35M/150M/650M + BALM-paired registry with PCA), `quantum_kernels.py` (minimal + expressive single-register kernels), `classical.py`, `evaluation.py` (learning curves + Spearman), `visualization.py`, and CLI scripts `extract_embeddings.py` / `run_experiments.py` / `generate_figures.py`.
- This serves as the **ablation/baseline arm** of the two-register design (single-register on antibody-only features) — a starting point to build the target modality and two-register circuit on top of, not a competing design.
- Mirrored the DMS datasets (~162 MB) to `s3://scrippsresearch-mshafiei-hackathon/`, readable by the whole hackathon account; download instructions in `data/README.md`.
- Known gaps to address: `scripts/extract_embeddings.py` concatenates heavy+light for all models (breaks BALM-paired, which needs separate chains — the library `embed_and_reduce` path is already correct); PennyLane not yet installed in the base env, so the quantum arm has not been run end-to-end.

---

### 2026-05-29 — Prediction target: regression on log₁₀(KD/M) (Benjamin + Claude)

**Decision: regression, not binary classification.**

- Predict **log₁₀(KD/M)** per dataset separately. In code: `target = -(log10(kd_nm) - 9)` = `-log10(kd_nm * 1e-9)`.
- Primary metric: Spearman ρ (rank correlation — robust to scale, directly answers "does the model rank variants correctly?").
- Secondary metric: RMSE on log₁₀(KD/M).
- Implementation: kernel ridge regression (KRR) or SVR with the quantum/classical kernel matrix. Both are in scikit-learn.
- **Never pool regression targets across datasets.** Phillips, Dailey, MAGMA-seq were measured by different labs on different instruments with different dynamic ranges (Phillips caps at 1000 nM; MAGMA-seq goes to 20,000 nM). Cross-dataset absolute KD comparison is meaningless. Evaluate each dataset separately; use Spearman ρ for any cross-dataset summary (rank-invariant to scale differences).
- Binary classification was rejected: arbitrary threshold required, introduces class imbalance (Phillips piles up at detection floor), and discards the quantitative signal we worked to curate.

---

### 2026-05-29 — Pipeline slides created (Benjamin + Claude)

Two PowerPoint slides generated in `figures/`:

| File | Scope | DR methods shown |
|------|-------|-----------------|
| `quantab_pipeline.pptx` | Hackathon | PCA only |
| `quantab_pipeline_v2.pptx` | Full paper | PCA (★ hackathon) · PLSR (◆ paper) · RandProj (∅ null) |

v2 organises the DR comparison as three swim-lane bands so each can be presented as a separate step. Same classifier arms (classical, two-register quantum, ablation) shown in each slide — applied identically across DR methods. Generated by `figures/make_pipeline_slide.py` and `figures/make_pipeline_slide_v2.py`; re-run either script to regenerate.

---

### 2026-05-29 — Embedding architecture and quantum circuit design (Benjamin + Claude)

**Key architectural decision: two modalities, two registers.**

Embeddings come from two separate models:
- Antibody: IgBERT or BALM on **paired VH+VL** (single embedding per antibody sample). Paired input is required — do not embed chains separately.
- Target: ESM2 on target protein sequence.

Pipeline: extract separately → z-score normalize each independently → separate PCA/PLSR per modality → classical arm concatenates; quantum arm uses a two-register circuit.

**Why two registers?** Affinity is an interaction property. The two-register quantum kernel operates in the tensor product space H_ab ⊗ H_target (2^(k_ab+k_target)-dimensional), which classical concatenation cannot represent without exponential cost. This is now the core scientific claim of the paper.

**Qubit budget:** (k_ab, k_target) pairs: (6,6)=12, (8,8)=16, (10,10)=20, (14,14)=28 total qubits. Sweet spot for the paper: around 20–28 total qubits, matching informative embedding dimensionality.

**DR decisions:** PCA per modality for hackathon. For paper: add PLSR (label-predictive) and random projection (null baseline) as second DR axis. Number of components determined by cross-validated Spearman ρ, not fixed in advance.

**Key ablation:** single-register quantum kernel on concatenated features vs. two-register — isolates tensor product benefit specifically.

**Skeptical note on qubit count:** more qubits only help if PCA components carry real signal. With ~3000 samples, informative dimensionality is likely 20–35 PCs per modality. Elbow on explained variance curve (once data arrives) should set the ceiling, not hardware ambition.

---

### 2026-05-29 — Infrastructure setup (Benjamin + Claude)

- AWS CLI installed, SSO configured with profile `bnemoz-scripps` on hackathon account `127696279288`
- Working directly on https://github.com/brineylab/QuantAb (no fork). Each team member has their own branch: `benjamin` (bnemoz), Mahdi and Terrence create their own.
- EC2 `c5.4xlarge` launched (`i-038ef1f119518297a`) with 100 GB EBS; SSH via EICE documented in CLAUDE.md
- GPU quota (G/VT instances) is 0 on account — needs organizer to request increase before switching to `g4dn.xlarge` + `lightning.gpu`
- Qubit plan updated: test **6, 10, and 14 qubits** (three points on scaling axis)
- Affinity dataset schema confirmed: CSV with `heavy chain seq`, `light chain seq`, `target prot seq`, `affinity (KD)` — plain AA strings, no numbering scheme
- Mahdi to update MEMORY.md with specific AbLM checkpoint combinations to test

---

### 2025-05-20 — Initial brainstorm (Benjamin)

**Starting point:** Hackathon at Scripps with AWS support. Initial idea was quantum simulation of antibody-antigen energy landscapes (option 1: VQE on molecular systems). Pivoted after discussion:

- 10 qubits is far too few for meaningful molecular simulation of amino acid interactions (~hundreds of qubits needed even for 2 residues in minimal basis)
- VQE on H₂/LiH is a learning exercise but doesn't connect to the antibody problem

**Three options discussed:**
1. VQE on a hydrogen bond model — educational, honest about toy scale
2. Quantum ML / quantum kernels on AbLM embeddings — novel, plays to group strengths
3. Hybrid classical-quantum pipeline — engineering contribution, lower ceiling

**Selected: Option 2** — highest novelty, leverages the group's unique assets (15 pretrained AbLMs at controlled scales), and the data-scarcity constraint actually sharpens rather than kills the research question.

**Key insight:** The Briney lab's scaling laws work provides a *controlled ladder of embedding quality* that nobody else has. This enables a two-dimensional scaling study (classical model scale × quantum circuit expressibility) that is original and systematic.

**Narrative arc:** "We characterized classical scaling (Shafiei Neyestanak et al. 2025). Now we probe whether quantum methods add a new scaling dimension."

**Publication venues discussed:** Nature Machine Intelligence (full paper), Nature Biotechnology (perspective), Quantum Science and Technology, JCIM (fallback).
