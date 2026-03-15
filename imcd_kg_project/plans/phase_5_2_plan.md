# Phase 5.2 Plan: Cell-Type Composite Node Fine-Tuning

## Goal
Use professor's cell-type-specific t-stat data to build composite intermediate
nodes in the RTX-KG2 graph, then fine-tune a GATConv model so that adalimumab
ranks highly for Castleman disease (iMCD-TAFRO) but not for unrelated diseases.

## Why This Should Work (vs Phase 5.1)
Phase 5.1 failed because 70 edge weights were swamped by 174K training pairs
and W_e was never trained to interpret fold-change values.

Phase 5.2 fixes both:
- Composite nodes make the cell-type signal STRUCTURAL (a new graph path),
  not just a weight on an existing edge. The model has to route through the
  new node, which forces it to learn the cell-type context.
- Fine-tuning on the iMCD subgraph with known treatments as supervision
  teaches the model that high t-stat paths correlate with drug relevance.

## Architecture: What "Composite Node" Means

Before (Phase 5.1):
  Castleman disease ---[weight=30.7]--> TNF --> Adalimumab

After (Phase 5.2):
  Castleman disease --> [TNF|T_cells node] --> TNF --> Adalimumab
                    --> [TNF|Monocytes node] --> TNF
                    --> [IL6|T_cells node]  --> IL6
                    ... (one node per top gene × cell type pair)

The composite node is a NEW node in the graph. It sits between the disease
and the canonical protein. The t-stat becomes the edge weight on the edge
FROM the disease TO the composite node.

The canonical protein node (TNF, IL6, etc.) KEEPS all its existing connections
to drugs and other proteins. Adalimumab still connects to TNF directly.
Nothing is broken.

## Data Available
- CSV: data/experimental/iMCD_TAFRO_cell_specific_tstats.csv
  - 12,500 genes × 5 cell types (B cells, ILC, Megakaryocytes/platelets,
    Monocytes, T cells)
  - Values are t-statistics (scRNA-seq, iMCD patients vs healthy)
  - Range: 0 to ~96.73 (needs normalization)
- TNF in data: Monocytes=32.22, T cells=2.91 (flag to Prof. Singh — paper
  says naive CD4+ T cells should be highest)

## Subphases

### 5.2.1 — Gene Symbol Mapping (LOCAL, testable now)
**What:** Map the 12,500 gene symbols in the CSV to UniProtKB IDs that exist
in RTX-KG2 (our graph). Not all genes will have matches.
**Why:** The graph uses UniProtKB IDs (e.g., UniProtKB:P01375 for TNF), not
gene symbols (e.g., "TNF"). We need to know how many genes from the CSV
actually exist as nodes in our graph before building anything.
**Output:** coverage_report.json — list of matched/unmatched genes, coverage %
**Test:** Unit test confirms TNF maps to UniProtKB:P01375, total coverage > 30%
**Files to create:**
  - src/enhanced_kgnn/gene_mapper.py
  - tests/test_gene_mapper.py

### 5.2.2 — Composite Node Builder (LOCAL, testable)
**What:** Given the gene→UniProtKB mapping, build a modified graph Data object
that includes composite (cell_type, gene) intermediate nodes.
**Why:** These nodes are the structural innovation. Each one represents
"this gene expressed in this cell type in iMCD patients."
**Output:** Extended PyG Data object with composite nodes added
**Test:** Verify node count increases by exactly N_composite, edge count increases
by 2×N_composite (one disease→composite, one composite→canonical protein)
**Files to create:**
  - src/enhanced_kgnn/composite_node_builder.py
  - tests/test_composite_node_builder.py

### 5.2.3 — T-Stat Normalization (LOCAL, testable)
**What:** Normalize the t-statistics before using them as edge weights.
T-stats range from 0 to 96.73 — too wide a range for attention mechanisms.
Strategy: clip at 99th percentile, then min-max scale to [0.1, 1.0].
Why NOT raw t-stats: a value of 96.73 vs 2.91 in the same softmax will
destroy the gradient just like 30.7x did in Phase 5.1.
**Output:** Normalized weight dictionary: {(gene, cell_type): weight}
**Test:** All output weights in [0.1, 1.0], TNF|T_cells maps to known value
**Files to create:**
  - src/enhanced_kgnn/tstat_normalizer.py
  - tests/test_tstat_normalizer.py

### 5.2.4 — Fine-Tuning Loop (SOCKEYE)
**What:** Fine-tune the pre-trained GATConv on the iMCD subgraph.
- Extract k-hop subgraph around Castleman disease node
- Add composite nodes (from 5.2.2) with normalized weights (from 5.2.3)
- Supervision: siltuximab + tocilizumab = positive pairs (known treatments)
- Adalimumab = held out, NOT in training, evaluated only
- 50-100 epochs, lr=1e-4 (smaller than pre-training to preserve weights)
**Output:** Fine-tuned model checkpoint in results/phase_5_2/
**Test:** Loss decreases, siltuximab/tocilizumab rank improves during fine-tune
**Files to create:**
  - src/enhanced_kgnn/finetuner.py
  - jobs/phase_5_2_finetune.sh
  - tests/test_finetuner.py (unit test on synthetic graph)

### 5.2.5 — Evaluation (SOCKEYE)
**What:** Rank all drugs for Castleman disease and for 3 non-TNF diseases.
Check: does adalimumab rank higher for Castleman than for diabetes/hypertension?
**Output:** results/phase_5_2/evaluation.json
**Test:** If adalimumab rank for Castleman < rank for non-TNF diseases, success.

## Open Questions (flag to Prof. Singh)
1. "T cells" in CSV — is this aggregated? Can we get naive CD4+ T cell
   breakdown specifically? TNF t-stat is 32.22 in Monocytes, only 2.91 in
   T cells, which contradicts the paper's main finding.
2. Threshold for composite nodes: include all 12,500 genes or top-N by t-stat?
   Recommend top 100-200 per cell type (t-stat > some threshold) to keep
   graph manageable.

## Confirmed Facts from Diagnostics (phase_5_2_diagnostics.py)
- Graph is DIRECTED. Use G.predecessors() for incoming edges.
- Castleman: 68 protein neighbors (all incoming), 51/68 in CSV
- Siltuximab CHEMBL1743070: direct edge to Castleman = True
- Tocilizumab CHEMBL1237022: direct edge to Castleman = True
- Adalimumab CHEMBL1201580: direct edge to Castleman = False (held out)
- Key path: Adalimumab->TNF->Castleman intact
- TNF t-stats: Monocytes=32.22, T cells=2.91
- Strong signals: NFKBIA, CD79A, IRF7, STAT1, CR1, VEGFA, IL1B, STAT3

## Current Status
- [x] 5.2.1 Gene mapping — COMPLETE (local, 21/21 tests passing)
      Files: src/enhanced_kgnn/gene_mapper.py, tests/test_gene_mapper.py
- [x] Diagnostics — COMPLETE (Sockeye, all 6 checks passed)
      File: jobs/phase_5_2_diagnostics.py
- [x] 5.2.2 Composite node builder — COMPLETE (local, 20/20 tests passing)
      Files: src/enhanced_kgnn/composite_node_builder.py
             tests/test_composite_node_builder.py
      Config: min_tstat=2.0, normalization clips at 99th pct, scales to [0.1,1.0]
- [x] 5.2.3 T-stat normalization — DONE (built into composite_node_builder.py)
- [x] 5.2.4 Fine-tuning loop — COMPLETE (local, 4/4 tests passing)
      Files: src/enhanced_kgnn/finetuner.py, tests/test_finetuner.py
             jobs/phase_5_2_finetune.sh, jobs/run_phase_5_2.py
      Also: gat_predictor.py patched to read edge 'weight' attributes
- [x] 5.2.5 Evaluation — COMPLETE (Sockeye, SLURM job ran successfully)
      Results: results/phase_5_2/phase_5_2_results.json

## Phase 5.2 Final Results (March 2026, Sockeye)

Seeds: [42, 123, 303], epochs=200, min_tstat=2.0, 49 composite nodes created.

| Target disease     | Adalimumab mean rank |
|--------------------|----------------------|
| Castleman (iMCD)   | **#13,346** (std=416)|
| Type 2 Diabetes    | #15,008              |
| Hypertension       | #14,928              |
| Alzheimer Disease  | #14,948              |

**Verdict: DISEASE-SPECIFIC** — adalimumab ranks ~1,600 positions higher
for Castleman than for non-TNF diseases (mean non-TNF rank = #14,961).

Interpretation:
- The composite node architecture successfully creates a structural path that
  the GATConv model routes through, giving Castleman a TNF-weighted signal.
- The ~1,600 rank difference is statistically consistent across 3 seeds (std=416).
- This is a meaningful result: Phase 5.1 showed ZERO disease-specificity
  (Castleman ranked WORSE with TNF weights). Phase 5.2 shows POSITIVE specificity.
- The ranking is still in the mid-teens thousands out of ~66,304 drugs, so
  absolute rank is not yet clinically useful — but the disease-specificity
  signal is now real and structural.
