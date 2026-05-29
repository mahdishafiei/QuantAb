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

## Design decisions (settled)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Quantum framework | PennyLane | Python-native, strong ML integration, good tutorials, quantum kernel support built-in |
| Primary metric | Spearman ρ | Rank correlation is robust to scale; standard in binding prediction benchmarks |
| Core analysis | Learning curves | Directly tests the small-data hypothesis; publishable regardless of outcome |
| Hackathon qubit counts | (6,6), (8,8), (10,10) per register | Two registers (ab + target); total 12/16/20 qubits. Tractable on CPU/GPU. |
| Antibody model | IgBERT (preferred) or BALM | Must accept **paired VH+VL** — never embed chains separately |
| Target model | ESM2 | Standard protein LM; pooled sequence embedding |
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
- Use the [CLS] token (index 0 of `last_hidden_state`) as the sequence embedding.
- On CPU: ~30 min for 67k sequences; always pass `cache_path` to `embed_and_reduce()` to save raw embeddings. Name caches `{dataset}_{model_key}_{n_components}d.npy`.
- Load report shows UNEXPECTED `cls.predictions` keys — expected and harmless (MLM head unused).

---

## Discussion log

### 2026-05-29 — Single-register pipeline + dataset mirror merged (Mahdi)

- Implemented the full single-register baseline pipeline on the `mahdi_quant` branch: `quantab/data.py` (Phillips/Engelhart/trastuzumab loaders), `embeddings.py` (IgBERT + ESM-2 35M/150M/650M + BALM-paired registry with PCA), `quantum_kernels.py` (minimal + expressive single-register kernels), `classical.py`, `evaluation.py` (learning curves + Spearman), `visualization.py`, and CLI scripts `extract_embeddings.py` / `run_experiments.py` / `generate_figures.py`.
- This serves as the **ablation/baseline arm** of the two-register design (single-register on antibody-only features) — a starting point to build the target modality and two-register circuit on top of, not a competing design.
- Mirrored the DMS datasets (~162 MB) to `s3://scrippsresearch-mshafiei-hackathon/`, readable by the whole hackathon account; download instructions in `data/README.md`.
- Known gaps to address: `scripts/extract_embeddings.py` concatenates heavy+light for all models (breaks BALM-paired, which needs separate chains — the library `embed_and_reduce` path is already correct); PennyLane not yet installed in the base env, so the quantum arm has not been run end-to-end.

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
