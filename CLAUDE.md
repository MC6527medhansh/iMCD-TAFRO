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
      gat_predictor.py                 — NEW GATConv approach (Phase 5, current)
  tests/
    test_gat_unit.py                   — 27 unit tests, all passing locally
  jobs/
    phase_5_1_gat_validation.sh        — Phase 5.1 SLURM job (completed)
    phase_4_2_specificity.sh           — Phase 4.2 SLURM job (completed, old approach)
  results/
    phase_5_1/                         — Phase 5.1 results (on Sockeye)
    phase_4_2_specificity/             — Phase 4.2 results (on Sockeye)
    diagnostics/                       — Jan 2026 diagnostic results (on Sockeye)
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

### What is BLOCKED

**Waiting on scRNA-seq data from Professor Singh.** This data will give:
- Fold changes for MANY genes (not just TNF) in iMCD patients vs healthy
- Cell-type-stratified expression (e.g., TNF in naive CD4+ T cells)
- Format expected: (gene_symbol, log2_fold_change) or (gene_symbol,
  fold_change, cell_type) — same 3-column format as the general tool design

This data is the direct input to the fine-tuning pipeline. Without it we
only have 1 gene (TNF, log2=4.94) which is too thin for robust fine-tuning.

### What CAN be built now (Phase 5.2 — not yet implemented)

**Fine-tuning pipeline on iMCD subgraph:**

Architecture:
1. Load pre-trained GATConv from base RTX-KG2 training (full_graph.pkl)
2. Extract iMCD-relevant subgraph (Castleman + k-hop neighborhood)
3. Attach disease edgelist as weighted edges (log2 fold changes)
4. Fine-tune using KNOWN Castleman treatments as positive supervision
   (siltuximab, tocilizumab — NOT adalimumab, which should emerge from
   the TNF fold-change signal without being explicitly supervised)
5. Adalimumab should rank higher because TNF edge weight is high,
   and adalimumab targets TNF — even without being in training pairs
6. Evaluate: adalimumab rank for Castleman vs rank for non-TNF diseases

**Key design decisions for Phase 5.2:**
- Use log2 fold change (4.94), not linear (30.7)
- Supervision signal: siltuximab + tocilizumab as positives (known IL-6 pathway)
- Adalimumab is evaluation-only — should emerge from fold-change signal
- Do NOT apply weight noise to structural ontology edges (biolink:subclass_of etc.)
- Only weight biological interaction edges (gene-disease, gene-drug, PPI)

**What changes when scRNA-seq data arrives:**
- The edgelist expands from 1 gene (TNF) to potentially 20-50 genes
- Each gene becomes a weighted edge in the subgraph
- Fine-tuning has richer signal = more robust rankings
- Adalimumab's emergence from fold-change becomes more convincing

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
