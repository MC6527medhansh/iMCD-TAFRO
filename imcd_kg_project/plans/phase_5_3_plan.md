# Phase 5.3 Plan: Per-Cell-Type Composite Node Experiments

## Goal

Two improvements over Phase 5.2 driven by Professor Singh's feedback:

1. Use ALL ~10,900 matched genes from the CSV (not just Castleman's 68
   direct graph neighbors). The t-stat IS the evidence of relevance.
   A gene does not need a pre-existing graph edge to Castleman.

2. Run one experiment per cell type so we can compare which cell type
   produces the strongest adalimumab ranking for Castleman. Professor's
   hypothesis: T cells should win because the paper's key finding is TNF
   upregulation in naive CD4+ T cells.

## Why Phase 5.2 Was Limited

Phase 5.2 only created composite nodes for genes that ALREADY had a
direct edge to Castleman in RTX-KG2. That gave 49 composite nodes.

The limitation: RTX-KG2's Castleman neighborhood reflects what was
known at graph construction time, not the full iMCD biology. The
scRNA-seq CSV has 12,500 genes — that IS the biology. We should let
the t-stat data decide what is relevant, not the existing graph topology.

## Architecture Change

Before (Phase 5.2):
  - Candidate genes = Castleman's 68 direct predecessors INTERSECT CSV genes
  - Result: 49 composite nodes

After (Phase 5.3):
  - Candidate genes = ALL genes in CSV that map to UniProtKB in graph (~10,900)
  - Result: potentially thousands of composite nodes per experiment
  - Same edge structure: Castleman -> [gene|cell_type] -> canonical protein

## Experiments

| Experiment            | cell_types param              | seeds        |
|-----------------------|-------------------------------|--------------|
| Combined              | None (all 5 cell types)       | [42, 123, 303] |
| B cells only          | ["B cells"]                   | [42]         |
| ILC only              | ["ILC"]                       | [42]         |
| Megakaryocytes only   | ["Megakaryocytes/platelets"]  | [42]         |
| Monocytes only        | ["Monocytes"]                 | [42]         |
| T cells only          | ["T cells"]                   | [42]         |

## Files Changed or Created

- src/enhanced_kgnn/composite_node_builder.py
    - Replaced _load_tstats_for_neighbors with _load_tstats_all_genes
    - Added cell_types param to build()
- src/enhanced_kgnn/finetuner.py
    - Added cell_types param to run()
    - cell_types saved in results JSON
- tests/test_composite_node_builder.py
    - Added TestAllGenesMode with 2 new tests (22 total, all passing)
- jobs/run_phase_5_3.py       — entry point, runs 6 experiments
- jobs/phase_5_3_celltype.sh  — SLURM job (48h, 64GB, 8 CPUs)

## Success Criterion

Adalimumab ranks better for Castleman than non-TNF diseases in at least
one cell-type-specific experiment. We expect T cells and Monocytes to
show the strongest signal given TNF t-stats of 2.91 and 32.22.

## Open Question (flag to Prof. Singh)

The "T cells" column in the CSV is an aggregate across all T cell
subtypes. TNF t-stat in this aggregate is only 2.91. The paper's
key finding is TNF upregulation specifically in naive CD4+ T cells,
which would be diluted in the aggregate. If a finer breakdown is
available, it would likely make T cells the dominant signal.

## Current Status

- [x] composite_node_builder.py updated — 22/22 tests passing locally
- [x] finetuner.py updated — 47/47 tests passing locally
- [x] run_phase_5_3.py created
- [x] phase_5_3_celltype.sh created
- [ ] Sockeye run — NEXT: commit, git pull on Sockeye, sbatch jobs/phase_5_3_celltype.sh
- [ ] Results analysis
