# PLAN.md — QuantAb Project Plan

## Motivation

Antibody language models (AbLMs) compress antibody sequences into rich embedding spaces that encode structural and functional information. Downstream prediction of binding affinity from these embeddings is constrained by the scarcity of labeled affinity datasets — typically hundreds to low thousands of measurements, not the millions available for pretraining.

Quantum kernel methods define similarity measures in exponentially large Hilbert spaces and are naturally suited to small-data regimes. The central question is whether these quantum-defined kernels capture nonlinear structure in AbLM embedding manifolds that classical kernels (RBF, polynomial, Matérn) do not.

## Research question

> In the small-labeled-data regime characteristic of antibody affinity prediction, do quantum kernel methods operating on antibody language model embeddings capture binding-relevant structure that classical kernels miss?

A secondary question, unique to our group:

> Does any quantum kernel advantage depend on the quality/scale of the classical embedding model?

This second question leverages the 15 pretrained AbLMs from Shafiei Neyestanak et al. (2025), spanning 5 model sizes × 3 data sizes, all publicly available.

## Experimental design

### Data

Use a public deep mutational scanning (DMS) dataset where a single antibody is systematically mutated and binding is measured for hundreds of variants. Candidate datasets (to be finalized):

- Taft et al. — DMS of therapeutic antibodies
- Adams et al. — DMS with binding measurements
- Phillips et al. (2021) — CR9114 anti-influenza DMS
- Mason et al. — SARS-CoV-2 DMS datasets

Selection criteria: sufficient variant count (>200), continuous binding measurement (KD or enrichment score), publicly available sequences.

### Features (embeddings)

1. Extract embeddings from Briney lab AbLMs at 3 scales (small, medium, large) from the scaling laws model collection
2. Optionally include ESM-2 and/or AbLang embeddings as external baselines
3. Dimensionality reduction via PCA to 4, 6, 8, and 10 dimensions (matching qubit budget)

### Quantum arm

- **Framework:** PennyLane (Xanadu) — Python-native, strong quantum ML and quantum chemistry support, integrates with PyTorch and scikit-learn
- **Encoding:** Angle encoding of reduced embeddings into quantum circuits
- **Kernels:** Quantum kernel estimation (inner products in Hilbert space) → feed kernel matrix into SVM
- **Circuit variants:** Test 2–3 ansatz architectures varying entanglement structure and depth
- **Simulation:** Statevector simulator on classical hardware (10 qubits is trivial to simulate)

### Classical arm (baselines)

Same reduced embeddings, same SVM framework:

- Linear kernel SVM
- RBF kernel SVM
- Polynomial kernel SVM
- Random forest (non-kernel baseline)
- Simple feedforward neural network (optional)

### Evaluation

- **Primary metric:** Spearman rank correlation with measured binding affinity
- **Key analysis: Learning curves** — plot performance vs. training set size (from 20 examples to full training split). This is the core of the small-data narrative.
- **Secondary analysis: Scaling surface** — performance as a function of (classical model scale) × (quantum circuit expressibility: qubits, depth)
- **Cross-validation:** k-fold CV, repeated with different random seeds

### Possible outcomes

| Outcome | Interpretation | Publishable? |
|---------|---------------|-------------|
| Quantum kernels help more with weaker embeddings | Quantum compensates for classical model limitations | Yes — democratization angle |
| Quantum kernels help more with stronger embeddings | Quantum needs rich features to exploit structure | Yes — quality floor finding |
| No meaningful difference anywhere | Ruled out rigorously across controlled grid | Yes — negative result with systematic evidence |

## Hackathon scope (minimum viable deliverable)

The hackathon is the proof of concept. Keep it focused:

1. **1 DMS dataset** (pick the cleanest one)
2. **3 AbLM checkpoints** (small, medium, large from scaling laws collection)
3. **PCA to 6 and 10 dimensions** (two qubit budgets)
4. **2 quantum kernel variants** (minimal and expressive ansatz)
5. **3 classical baselines** (linear SVM, RBF SVM, random forest)
6. **Learning curves** for all methods
7. Visualization notebook with results

## Paper scope (post-hackathon)

- Full 15-model grid (5 sizes × 3 data sizes)
- Multiple DMS datasets
- 3+ quantum circuit architectures with systematic depth/entanglement variation
- Dimensionality sweep: 4, 6, 8, 10 qubits
- Statistical analysis of scaling surfaces
- Resource estimation: how many qubits for practical advantage?

## Target venues

- **Full paper:** Nature Machine Intelligence, Quantum Science and Technology
- **Perspective/opinion:** Nature Biotechnology ("Quantum computing for antibody engineering — where are we really?")
- **Fallback:** Journal of Chemical Information and Modeling, Bioinformatics

## Narrative arc

"We characterized how classical AbLM performance scales with data and parameters (Shafiei Neyestanak et al. 2025). Now we probe whether quantum methods introduce a new scaling dimension — and whether the answer depends on where you sit on the classical scaling curve."
