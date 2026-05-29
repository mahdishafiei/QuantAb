# MEMORY.md — Design Decisions & Discussion Log

## Settled decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | IgBERT (`Exscientia/IgBert`) | HuggingFace model, loads in 2 lines, no checkpoint wrangling, 1024-dim output |
| Quantum framework | PennyLane + `lightning.qubit` | Python-native, fast CPU statevector simulator |
| Primary metric | Spearman ρ | Rank correlation; robust to scale; standard in binding benchmarks |
| Core analysis | Learning curves | Directly tests the small-data hypothesis; publishable regardless of outcome |
| Dimensionality reduction | PCA to 6 and 10 dims | Matches qubit budget; interpretable; same projection for quantum + classical |
| Datasets | All available (Phillips primary) | Phillips 2021 CR9114 is large (~67k), paired, clean KD values |
| Classical baselines | Linear/RBF/Poly SVM + Random Forest | Standard kernel zoo + non-kernel reference |
| Noise model | Ideal statevector simulation | Cleaner science; noise analysis deferred to full paper |
| Approach to results | Benchmark, not advocacy | Negative results are equally valuable |

---

## Dataset decisions

**Primary:** Phillips et al. 2021 CR9114 — ~67k variants with full paired heavy+light sequences
and continuous -log(KD) measurements against H1 and H3 influenza strains.
Path: `/home/jovyan/work/Hackaton/DMS_Data/02_phillips_cr9114_cr6261/`

**Secondary (loaded):**
- Engelhart AAYL: `05_engelhart_aayl/` — full paired sequences + binding score
- Shanehsazzadeh trastuzumab: `06_shanehsazzadeh_trastuzumab/` — HCDR3 variants + KD (nM)

**Deferred (complex formats):**
- Dailey CR9114: single-point mutation table, no full sequences
- MAGMA-seq: complex multi-file format
- Adams titeseq: PDB-level data

---

## IgBERT notes

- Model ID: `Exscientia/IgBert`
- Hidden dim: 1024 (BERT-large scale, not base)
- Input format: space-separated amino acids — `"E V Q L V E S ..."`
- Use [CLS] token (index 0 of `last_hidden_state`) as sequence embedding
- On CPU: ~30 min for 67k sequences; use `cache_path` in `embed_and_reduce()` to save raw embeddings
- Load report shows UNEXPECTED cls.predictions keys — expected, harmless (MLM head not used)

---

## Open questions for paper (not hackathon)

- Does quantum advantage appear only at very small N (< 50 training examples)?
- Does the advantage depend on embedding quality / model scale?
- Is 10 qubits enough to express meaningful structure in IgBERT's 1024-dim space?
- Paired heavy+light vs heavy-only input to IgBERT — does pairing help?
- Real quantum hardware (IBM Quantum, Amazon Braket) — resource estimation?

---

## Discussion log

### 2025-05-20 — Initial project design (Benjamin)

Three options evaluated:
1. VQE on hydrogen bond model — educational but disconnected from antibody problem
2. **Quantum ML / quantum kernels on AbLM embeddings** ← selected
3. Hybrid classical-quantum pipeline — engineering, lower ceiling

Rationale for option 2: highest novelty, leverages Briney lab's unique asset of 15 pretrained AbLMs
at controlled scales, and data scarcity actually sharpens the research question.

Narrative arc: "We characterized classical AbLM scaling (Shafiei Neyestanak et al. 2025).
Now we probe whether quantum methods add a new scaling dimension."

### 2026-05-29 — Hackathon kickoff

- Switched from custom AbLM checkpoints to IgBERT for embedding extraction (simpler, reproducible).
- Custom AbLMs deferred to full paper where scale comparison can be done systematically.
- Forked repo to mahdishafiei/QuantAb; all work will be done from the fork.
- Full pipeline scaffold implemented: data loaders, IgBERT extractor, quantum + classical kernels,
  evaluation, visualization.
- 67,455 variants loaded across 5 datasets. End-to-end smoke test passed.
