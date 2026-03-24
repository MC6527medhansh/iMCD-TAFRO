# CLAUDE.md — iMCD-TAFRO Project Context

Read this at the start of every session. It contains everything needed to
continue work without re-explaining from scratch.

---

## Who and Where

- Developer: Medhansh Choubey (CWL: mchoubey), Prof. Amrit Singh's lab, UBC
- Local: Mac, repo at `/Users/medhanshchoubey/iMCD-TAFRO/`
- Sockeye HPC: `/scratch/st-singha53-1/mchoubey/iMCD-TAFRO/`
- Sockeye SLURM account: `st-singha53-1` (CPU), `st-singha53-1-gpu` (GPU)
- GitHub: `MC6527medhansh/iMCD-TAFRO` (PAT embedded in remote URL on Sockeye)
- Workflow: always `git pull` then run. Never rsync.

---

## The Big Picture (Professor Singh's Vision)

Build a generalizable drug repurposing pipeline inspired by TxGNN
(Huang et al., Nature Medicine 2024). The idea:

1. Train a GNN once on RTX-KG2 (a large biomedical knowledge graph)
2. Any researcher brings an edgelist: (disease_name, gene_symbol, fold_change)
3. Those genes get injected as weighted edges into the graph
4. The model outputs a ranked drug list for that disease

iMCD-TAFRO is the proof of concept. Heart failure (HF-KG) is the second
validation case. Once iMCD works cleanly, the repo becomes the general tool
with good documentation.

---

## Project 1: iMCD-TAFRO (Primary — Active)

### What iMCD-TAFRO Is

iMCD-TAFRO is a rare inflammatory disease. The key biological finding from
the paper: TNF is upregulated 30.7x (log2 fold change = 4.94) specifically
in naive CD4+ T cells. Adalimumab is a TNF inhibitor that should — in theory
— treat this disease. The goal is to have the model rank adalimumab highly
for Castleman disease specifically, not for every disease.

### Critical Entity IDs (hardcoded throughout)

- Adalimumab: `CHEMBL.COMPOUND:CHEMBL1201580`
- Castleman disease: `MONDO:0015564`
- TNF protein: `UniProtKB:P01375`
- TNF log2 fold change: 4.94 (linear: 30.7x)

### Graph Facts (confirmed from Sockeye runs)

- Full graph: 160,936 nodes, 3,626,585 edges (7,253,170 bidirectional)
- Node types: 66,304 drugs, 21,663 diseases, 72,969 proteins/other
- Adalimumab -> TNF: 1 hop direct edge (confirmed)
- TNF -> Castleman: 1 hop direct edge (confirmed)
- Castleman has 70 direct protein neighbors
- Graph stored at: `data/processed/full_graph.pkl` (on Sockeye scratch only)
- Training data: `data/kgml_data/training_data/` (RepoDB, semmed, ndf, mychem)
- Training pairs loaded: 174,263 (138,129 positive, 36,134 negative)
- No data leakage: adalimumab-Castleman pair NOT in training data

### Key Files

```
imcd_kg_project/
  src/
    config.py                          — paths + entity IDs + hyperparams
    enhanced_kgnn/
      enhanced_predictor.py            — OLD GraphSAGE approach (kept for reference)
      gat_predictor.py                 — GATConv approach (Phase 5, active)
      gene_mapper.py                   — maps gene symbols → UniProtKB IDs (Phase 5.2)
      composite_node_builder.py        — builds (gene|cell_type) intermediate nodes (Phase 5.2)
      finetuner.py                     — full training + evaluation pipeline (Phase 5.2)
  tests/
    test_gat_unit.py                   — 32 unit tests (27 + 5 TestGetDrugRanks, Phase 5.5)
    test_gene_mapper.py                — 21 tests (Phase 5.2)
    test_composite_node_builder.py     — 20 tests (Phase 5.2)
    test_finetuner.py                  — 6 tests (Phase 5.5: added test_drug_ranks_in_results)
  jobs/
    phase_5_5_drugtable.sh             — Phase 5.5 SLURM job (NEXT to run)
    run_phase_5_5.py                   — entry point for Phase 5.5 (6 experiments)
    phase_5_4_notstat.sh               — Phase 5.4 SLURM job (COMPLETE)
    run_phase_5_4.py                   — entry point for Phase 5.4 job
    phase_5_3_celltype.sh              — Phase 5.3 SLURM job (COMPLETE)
    run_phase_5_3.py                   — entry point for Phase 5.3 job
    phase_5_2_finetune.sh              — Phase 5.2 SLURM job (COMPLETE)
    run_phase_5_2.py                   — entry point for Phase 5.2 job
  data/
    experimental/
      iMCD_TAFRO_cell_specific_tstats.csv  — 12,500 genes × 5 cell types (Prof. Singh's data)
  results/
    phase_5_4/                         — min_tstat=0.0 results (on Sockeye)
    phase_5_3/                         — per-cell-type results (on Sockeye)
    phase_5_2/                         — DISEASE-SPECIFIC result (on Sockeye)
    phase_5_1/                         — Phase 5.1 results (on Sockeye)
    phase_4_2_specificity/             — Phase 4.2 results (on Sockeye)
    diagnostics/                       — Jan 2026 diagnostic results (on Sockeye)
  plans/
    phase_5_5_plan.md                  — Phase 5.5 drug ranking table plan
    phase_5_4_plan.md                  — Phase 5.4 full analysis + results
    phase_5_3_plan.md                  — Phase 5.3 per-cell-type plan
    phase_5_2_plan.md                  — full subphase breakdown + final results
```

### Local conda env

- Name: `imcd_kg`
- Run tests: `KMP_DUPLICATE_LIB_OK=TRUE conda run -n imcd_kg python -m pytest tests/test_gat_unit.py -v`
- PyG version local: 2.6.1, Python 3.9
- PyG version Sockeye: 2.7.0, PyTorch 2.9.0+cpu, Python 3.10 (no CUDA in env)

---

## Full Experimental History

### Phase 1-3: Data loading and baseline (Complete)

Built full RTX-KG2 loader, entity verification, graph construction. Validated
all critical entities exist. RepoDB loaded as supervision labels.

### Phase 4: GraphSAGE + Node Feature Injection (FAILED — do not revisit)

**What was tried:** Injected TNF log2 fold change (4.94) as a 4th scalar
feature on the adalimumab and TNF nodes. Compared baseline (3D features) vs
enhanced (4D features).

**Why it failed (confirmed by Phase 4.2 and Jan 2026 diagnostics):**
- Adalimumab-Castleman cosine similarity = 0.076 even with enhancement
- Adalimumab-TNF similarity = 0.897 (high, but wrong — scoring is drug-DISEASE
  dot product, not drug-protein)
- Enhanced rank ranged #4,780 to #65,777 across 5 seeds (range = 60,997)
  vs baseline range of only 1,417. Completely unstable.
- Phase 4.2 specificity test: non-TNF diseases improved MORE than TNF diseases
  (fold difference = 0.88). Not disease-specific at all.
- Root cause: GraphSAGE mean aggregation dilutes signal over 160K nodes.
  Injecting a feature on a node doesn't propagate meaningfully through the graph.

**Do not go back to this approach.**

### Phase 5.1: GATConv + Edge Weights (Complete — partial failure, important findings)

**What was tried:** Switched from GraphSAGE to GATConv (which natively supports
edge_attr). Added edge weight attributes to the 70 Castleman protein neighbor
edges. Three experiments:
- Exp A: GATConv + uniform weights (1.0) — new baseline
- Exp B: GATConv + random weights (0.5-5.0) on Castleman edges
- Exp C: GATConv + TNF fold-change weight (30.7) on TNF-Castleman edge

**Results:**
```
Exp A (uniform):  Adalimumab rank for Castleman: #14,064 mean
Exp B (random):   Adalimumab rank for Castleman: #14,556 mean (ranked WORSE)
Exp C (TNF 30.7): Adalimumab rank for Castleman: #15,275 mean (ranked WORSE)
```
The disease-specific "signal" in Exp C was 76 rank positions — pure noise.
Loss curves were nearly identical across all three experiments.

**Why it failed (root cause, fully diagnosed):**

1. Signal dilution: 70 weighted edges out of 7,253,170 = 0.001%. The RepoDB
   training gradient over 174,263 pairs across all diseases completely swamps
   the 70 Castleman edge weights. The model never sees a meaningful gradient
   from those edges.

2. W_e is untrained: GATConv with edge_dim=1 has a projection matrix W_e that
   maps edge features to attention logits. The model was trained on unit-weight
   edges. W_e never learned what "high weight = more important neighbor" means.
   Feeding 30.7 (or 4.94) into an untrained W_e produces garbage attention.
   The training objective (RepoDB pairs) provides zero gradient signal that
   correlates edge weight with drug efficacy. So W_e learns to treat weights
   as noise, not signal.

3. NOT a code bug: Unit tests confirm GATConv reads edge_attr correctly on
   small synthetic graphs (test_high_weight_changes_embedding passes). The
   issue is architectural/training, not implementation.

**The softmax trap (additional nuance):** PyG's GATConv concatenates the
projected edge feature (W_e * e_ij) to the attention logit, not multiplies.
The attention formula is:
  e_ij = LeakyReLU(a^T [Wh_i || Wh_j || W_e * edge_attr_ij])
A value of 30.7 fed through a randomly-initialized W_e is unpredictable.
Log2 transform (4.94) is always safer and more appropriate.

---

## Current Status and Next Steps

### Phase 5.2: Composite Node Fine-Tuning (COMPLETE — March 2026)

**What was built:**
- `src/enhanced_kgnn/gene_mapper.py` — maps 12,500 gene symbols → UniProtKB IDs (87% coverage)
- `src/enhanced_kgnn/composite_node_builder.py` — inserts (gene|cell_type) intermediate nodes
- `src/enhanced_kgnn/finetuner.py` — full training pipeline with siltuximab/tocilizumab supervision
- `jobs/phase_5_2_finetune.sh` + `jobs/run_phase_5_2.py` — SLURM job
- `gat_predictor.py` — patched to read 'weight' attribute from NetworkX edges

**Data:**
- `data/experimental/iMCD_TAFRO_cell_specific_tstats.csv` — 12,500 genes × 5 cell types
- T-stats range 0–96.73 (NOT bounded). Normalized: clip at 99th pct, min-max to [0.1, 1.0]
- TNF: Monocytes=32.22, T cells=2.91 (paper says naive CD4+ T cells highest — discrepancy, flag to Prof. Singh)
- Graph is DIRECTED. Use G.predecessors() for protein→disease edges.
- Castleman: 68 incoming protein edges. 51/68 have t-stats. 49 composite nodes at min_tstat=2.0.
- Confirmed CHEMBL IDs: Siltuximab=CHEMBL1743070, Tocilizumab=CHEMBL1237022, Adalimumab=CHEMBL1201580

**Results (Sockeye, seeds=[42,123,303], epochs=200):**
```
Adalimumab rank for Castleman:  #13,346 (std=416)
Non-TNF Type 2 Diabetes:        #15,008
Non-TNF Hypertension:           #14,928
Non-TNF Alzheimer Disease:      #14,948
Verdict: DISEASE-SPECIFIC
```
Adalimumab ranks ~1,600 positions better for Castleman than for non-TNF diseases.
Phase 5.1 had zero specificity (ranked WORSE with weights). Phase 5.2 is the first
positive disease-specificity signal in the project.

**Why it worked:** Composite nodes create a STRUCTURAL path through the graph, not
just a weight on an existing edge. The model routes through (TNF|Monocytes) and
(TNF|T_cells) composite nodes, which carry normalized t-stat weights — forcing it
to learn that high-t-stat paths correlate with Castleman drug relevance.

### Phase 5.3: Per-Cell-Type Composite Node Experiments (IN PROGRESS — local complete)

**Two changes from Phase 5.2, driven by Prof. Singh:**

1. ALL ~10,900 matched genes from CSV are used (Phase 5.2 used only 68 direct
   Castleman neighbors). The t-stat IS the evidence of relevance. A pre-existing
   graph edge to Castleman is no longer required.

2. 6 experiments: combined (all cell types, seeds=[42,123,303]) + one per cell
   type (B cells, ILC, Megakaryocytes/platelets, Monocytes, T cells; seeds=[42]).

**Files created/changed:**
- `src/enhanced_kgnn/composite_node_builder.py` — `_load_tstats_all_genes` replaces
  `_load_tstats_for_neighbors`; `build()` now takes `cell_types` param
- `src/enhanced_kgnn/finetuner.py` — `run()` now takes `cell_types` param
- `jobs/run_phase_5_3.py` — runs 6 experiments, saves to `results/phase_5_3/`
- `jobs/phase_5_3_celltype.sh` — SLURM job (48h, 64GB)
- `tests/test_composite_node_builder.py` — 22 tests (2 new), all passing
- `plans/phase_5_3_plan.md` — full plan

**Local test status: 48/48 passing**

**Sockeye results (job 8638905, March 15 2026):**
```
Combined (all cell types):   Castleman #13,042  non-TNF mean #14,485  gap=+1,443  SPECIFIC
Monocytes only:              Castleman #13,397  non-TNF mean #13,475  gap=+78     barely
T cells only:                Castleman #13,412  non-TNF mean #13,452  gap=+40     barely
B cells only:                Castleman #13,412  non-TNF mean #13,452  gap=+40     barely
ILC only:                    Castleman #13,333  non-TNF mean #13,356  gap=+23     barely
Megakaryocytes only:         Castleman #13,866  non-TNF mean #13,818  gap=-48     NOT SPECIFIC
```
Combined is the only experiment with a meaningful gap. Individual cell types at 1 seed
are too noisy. T cells and B cells identical rank despite different node counts — confirms
single-seed per-cell-type is insufficient to distinguish them. Professor's T cells > Monocytes
hypothesis not confirmed — supports argument that "T cells" aggregate dilutes naive CD4+ signal.
Phase 5.3 combined (#13,042) beats Phase 5.2 (#13,346) — all-gene approach confirmed better.

### Phase 5.4: All T-Stats Experiment (COMPLETE — March 2026)

**One change from Phase 5.3:** `MIN_TSTAT = 0.0` instead of 2.0. All non-zero
t-stats from the CSV are included. Exact zeros are still excluded (handled by
`_load_tstats_all_genes` which filters `tstat > 0` before min_tstat is applied).

**Files created:**
- `jobs/run_phase_5_4.py` — identical to run_phase_5_3.py except MIN_TSTAT=0.0
  and results go to results/phase_5_4/
- `jobs/phase_5_4_notstat.sh` — SLURM job (48h, 64GB)

**Sockeye results (seeds=[42,123,303] for combined, seed=42 for per-cell-type):**
```
Combined (all cell types):   Castleman #12,272  non-TNF mean #13,780  gap=+1,508  SPECIFIC  (best ever)
Megakaryocytes only:         ~1,300 nodes,  gap=+93   SPECIFIC
B cells only:                ~5,800 nodes,  gap=+16   barely
ILC only:                    ~7,200 nodes,  gap=-12   NOT SPECIFIC
Monocytes only:              ~9,500 nodes,  gap=-46   NOT SPECIFIC
T cells only:                10,208 nodes,  gap=-55   NOT SPECIFIC (worst)
```
Combined is the best result across all phases: rank #12,272 (best), gap +1,508 (largest).
T cells and Monocytes flipped to NOT SPECIFIC — lowering the threshold to 0.0 adds
thousands of low-t-stat genes that outweigh the true TNF signal for individual cell types.
The combined experiment survives because aggregating 5 cell types provides enough genuine
signal. Professor's T cells > Monocytes hypothesis not confirmed.
See `plans/phase_5_4_plan.md` for full analysis.

### Phase 5.5: Drug Ranking Table (IN PROGRESS — ready for Sockeye)

**Professor's question:** "I'm more interested in where existing drugs of
Castleman's are being ranked — anti-IL6, TNF, IL2. How does the ranking of
these three drugs differ between the cell types?"

Phases 5.2–5.4 only reported adalimumab's rank. Phase 5.5 extends to the
full drug ranking table: explicit rank + score for all confirmed drugs of
interest (adalimumab, siltuximab, tocilizumab) across all 6 experiments.

**One change from Phase 5.4:** the evaluation pipeline, not the parameters.
`min_tstat=0.0`, same 6 experiments, same seeds, same epochs.

**Implementation (already done):**
- `gat_predictor.py`: added `get_drug_ranks()` — single forward pass returning
  BOTH top-N table (generic) AND explicit tracking for drugs_of_interest
  regardless of rank position. Adalimumab at #13,000 is always tracked.
  Removed `get_top_n_drugs()` (was returning random compounds — wrong approach).
- `finetuner.py`: `DRUGS_OF_INTEREST` dict (3 confirmed CHEMBL IDs only).
  Seed loop calls `get_drug_ranks()` once. Saves `top_500_castleman`,
  `drug_ranks_castleman`, and aggregated `drug_mean_ranks_castleman`.
- Tests: `TestGetDrugRanks` (5 tests in `test_gat_unit.py`),
  `test_drug_ranks_in_results` + updated `test_results_saved_to_disk`
  (in `test_finetuner.py`).

**Confirmed drugs of interest:**
- Adalimumab:  CHEMBL1201580  (Anti-TNF, held out)
- Siltuximab:  CHEMBL1743070  (Anti-IL-6, positive supervision)
- Tocilizumab: CHEMBL1237022  (Anti-IL-6R, positive supervision)

Do NOT add unverified CHEMBL IDs. Search graph node names on Sockeye first.

**Why the first Phase 5.5 run (job 9338927) was wrong:**
Used top-200 generic list. Known drugs (at #13,000+) not in top 200 → not
reported. Fixed by replacing `get_top_n_drugs()` with `get_drug_ranks()`.

**Expected output:** siltuximab/tocilizumab at rank #1/#2 (in supervision),
adalimumab at ~#12,000–14,000 (held out), results differ slightly from Phase
5.4 due to CPU non-determinism.

**Next step: run tests locally, then commit and sbatch.**

**Progression across phases (combined experiment):**
- Phase 5.2: #13,346  gap ~1,600  (49 composite nodes, min_tstat=2.0)
- Phase 5.3: #13,042  gap +1,443  (~7,362 nodes, min_tstat=2.0, all genes)
- Phase 5.4: #12,272  gap +1,508  (32,118 nodes, min_tstat=0.0)  ← current best
- Phase 5.5: TBD      gap TBD     (32,118 nodes, min_tstat=0.0, drug table)

### Confirmed entity IDs (verified by name search in graph)
- Siltuximab:   CHEMBL.COMPOUND:CHEMBL1743070 (direct edge to Castleman: YES)
- Tocilizumab:  CHEMBL.COMPOUND:CHEMBL1237022 (direct edge to Castleman: YES)
- Adalimumab:   CHEMBL.COMPOUND:CHEMBL1201580 (direct edge to Castleman: NO — held out)
- Castleman disease: MONDO:0015564
- TNF protein: UniProtKB:P01375

### How this connects to Professor Singh's vision

The generalizable tool pipeline:
```
Any researcher's edgelist (disease, gene, fold_change)
        |
        v
[Pre-trained GATConv on RTX-KG2]  ← train once, reuse
        |
        v
[Fine-tune on disease subgraph]   ← short, disease-specific
        |
        v
[Ranked drug list]
```

iMCD is step 1 of proving this works. HF-KG is step 2. Once both work,
the generalizable tool is essentially iMCD with good documentation.

True zero-shot (no fine-tuning) would require training on fold-change data
from 50+ diseases simultaneously — a data problem we don't have yet.
Fine-tuning is the honest, achievable proof of concept right now.

---

## Project 2: HF-KG (Secondary — not currently active)

Repo: `MC6527medhansh/hf-kg-drug-repurposing` (private)
Sockeye: `/scratch/st-singha53-1/mchoubey/hf-kg-drug-repurposing/`
Conda env: `hf_kg`

Status: Phase 4 complete (GraphSAGE + contrastive loss, subtype differentiation
working). Carvedilol ranks #18 HFrEF, #164 HFpEF. Next: upgrade to GATConv,
add comorbidity subtypes, use fold-change edge weights.

Not active until iMCD proof of concept is done.

---

## Important Conventions

- Never commit without asking user first — always provide commit message for
  user to run themselves
- No auto-push
- Test with `KMP_DUPLICATE_LIB_OK=TRUE conda run -n imcd_kg python -m pytest`
- All new code in `src/enhanced_kgnn/` — keep gat_predictor.py as the active file
- enhanced_predictor.py is kept for reference only — do not modify
- SLURM jobs: 48GB RAM, 8 CPUs is sufficient for CPU runs on full graph
- Run tests locally before submitting to Sockeye
- config.py on Sockeye had a merge conflict (trivially resolved March 2026)
  — both sides had same content, just a comment difference

---

## Known Open Questions

1. **Reproducibility issue:** Identical code + seeds produced dramatically
   different rankings between October 2025 and January 2026. Not yet diagnosed.
   Do not ignore this when evaluating results.

2. **CUDA:** PyTorch on Sockeye is 2.9.0+cpu (no CUDA). GPU partition is
   available but needs a CUDA-enabled PyTorch install. Not set up yet.

3. **torch-scatter / torch-sparse warnings:** These appear on Sockeye but do
   not affect functionality. GATConv works without them in PyG 2.7.0.
