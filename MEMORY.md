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

**Action:** Need to survey these, check data availability, variant count, and measurement quality before committing.

---

### Which AbLM checkpoints?

The scaling laws paper produced 15 models (5 sizes × 3 data sizes). For the hackathon we need 3.

**Proposed selection:**
- Smallest model / largest data
- Medium model / largest data
- Largest model / largest data

Rationale: fix data size, vary model capacity — cleanest comparison axis. But should we instead fix model size and vary data? Or pick along the Pareto frontier?

**Action:** Discuss with Mahdi — he knows the models best.

---

### Embedding layer selection

Which layer(s) of the AbLM to extract embeddings from?
- Last hidden state (most common)?
- Pool across layers?
- CDR-only tokens vs. full sequence?

CDR-only might be more relevant for binding but reduces token count. Need to decide before extraction.

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

**Decision for hackathon:** Start with 2 architectures (one simple, one expressive). Systematic depth sweep in the paper.

---

### Noise simulation

Should we simulate quantum noise (shot noise, gate errors, decoherence) or use ideal statevector simulation?

- Ideal simulation is cleaner for the scientific question (does the Hilbert space structure help?)
- Noisy simulation is more realistic for near-term hardware claims

**Decision for hackathon:** Ideal simulation. Add noise models in the paper as a separate analysis.

---

### AWS compute allocation

Hackathon provides extended EC2 access. For 10-qubit statevector simulation, even a laptop is fine. But:
- Embedding extraction from large AbLMs may need GPU instances
- Full grid search (15 models × multiple DMS datasets × multiple circuits) could benefit from parallelization
- Any interest in testing on actual quantum hardware (IBM Quantum, Amazon Braket)?

---

## Design decisions (settled)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Quantum framework | PennyLane | Python-native, strong ML integration, good tutorials, quantum kernel support built-in |
| Primary metric | Spearman ρ | Rank correlation is robust to scale; standard in binding prediction benchmarks |
| Core analysis | Learning curves | Directly tests the small-data hypothesis; publishable regardless of outcome |
| Hackathon qubit counts | 6 and 10 | Two points on the scaling axis; 10 is tractable on a simulator |
| Classical baselines | Linear/RBF/Polynomial SVM + Random Forest | Standard kernel zoo plus a non-kernel reference |
| Approach to results | Benchmark, not advocacy | We report what we find. Negative results are valuable. |

---

## Discussion log

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
