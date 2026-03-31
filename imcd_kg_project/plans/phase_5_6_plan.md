# Phase 5.6 Plan: Finer Cell Type Resolution (42 cell types, abs t-stat)

## Goal

Rerun the 6-experiment drug ranking pipeline using Prof. Singh's new
42-cell-type CSV (`iMCD_TAFRO_cell_specific_tstats 2.csv`) with two
corrections:

1. **abs(t-stat)**: include all differentially expressed genes regardless
   of direction (up or down in flare). Previously the code filtered `tstat > 0`,
   discarding 238,359 gene-cell_type pairs (29% of non-zero entries).
   Prof. Singh asked explicitly why negative values were excluded.

2. **Finer cell type resolution**: 42 cell types vs the 5 broad types used
   in Phases 5.3–5.5. 7 cell types have extreme/broken t-stat values and
   are excluded. 35 clean cell types are used.

## Why We Re-run

New CSV + abs(tstat) change means the composite graphs are different from
all previous phases. Cannot post-process prior results.

## What Changed from Phase 5.5

| | Phase 5.5 | Phase 5.6 |
|---|---|---|
| CSV | iMCD_TAFRO_cell_specific_tstats.csv (5 cell types) | iMCD_TAFRO_cell_specific_tstats 2.csv (42 cell types) |
| T-stat type | log-fold change (mean flare − mean remission) | t-statistic (log-FC / variance), remission as reference |
| Sign filter | `tstat > 0` | `abs(tstat) > 0` |
| Cell types | 5 (B cells, ILC, Megakaryocytes, Monocytes, T cells) | 35 clean (7 excluded) |
| min_tstat | 0.0 | 0.0 |
| Epochs | 200 | 200 |
| Seeds (combined) | [42, 123, 303] | [42, 123, 303] |

## Why abs(t-stat)

In the new file, t-stats are computed with remission as the reference group.
A gene with a **positive** t-stat is up-regulated in FLARE.
A gene with a **negative** t-stat is down-regulated in FLARE (i.e., up in
remission). Both directions indicate that the gene is differentially
expressed in the disease state — both are biologically relevant as evidence
of cell-type involvement. Filtering to `tstat > 0` was discarding the
negative-direction evidence.

The composite node edge weight is `norm(abs(tstat))`. The direction is
encoded nowhere in the graph — only the magnitude of differential expression
matters for routing information toward Castleman.

## The New CSV

- File: `data/experimental/iMCD_TAFRO_cell_specific_tstats 2.csv`
- Genes: 23,373
- Cell types: 42 total, 35 clean, 7 broken (see below)
- Values: t-statistics = log-fold-change / variance
- Reference group: remission (+ = up in FLARE)
- TNF values differ from old CSV because the old file had raw log-fold
  change (TNF = +32.22 in Monocytes). New file is a t-stat: TNF = -0.05
  in Monocytes (the "Monocytes" column now maps to a different cell
  population in the finer annotation).

## Excluded Cell Types (7 — extreme values)

Confirmed in both file 1 and file 2. Max |value| in first 200 rows:

| Cell Type | Max |value| |
|-----------|-------------|
| Double-positive thymocytes | 8,594,777 |
| Alveolar macrophages | 1,328,824 |
| T(agonist) | 1,054,866 |
| Transitional NK | 928,783 |
| Memory CD4+ cytotoxic T cells | 657,331 |
| Double-negative thymocytes | 538 |
| CD8a/a | 282 |

These are numerical artifacts (likely division by near-zero variance). They
would dominate the normalization clip and produce garbage composite node weights.

## Clean Cell Types (35)

Age-associated B cells, B cells, CD16- NK cells, CD16+ NK cells,
CD8a/b(entry), Classical monocytes, CRTAM+ gamma-delta T cells, DC1, DC2,
Follicular helper T cells, HSC/MPP, ILC3, Macrophages, MAIT cells,
Megakaryocytes/platelets, Memory B cells, MNP, Monocytes, Myelocytes,
Naive B cells, NK cells, NKT cells, Non-classical monocytes, pDC,
Plasma cells, Regulatory T cells, Tcm/Naive cytotoxic T cells,
Tcm/Naive helper T cells, Tem/Effector helper T cells,
Tem/Temra cytotoxic T cells, Tem/Trm cytotoxic T cells,
Transitional B cells, Treg(diff), Trm cytotoxic T cells,
Type 17 helper T cells.

## TNF abs(t-stat) Across 35 Clean Cell Types

Ranked by strength (relevant for understanding which cell types carry TNF signal):

| Rank | Cell Type | TNF t-stat | Direction |
|------|-----------|------------|-----------|
| 1 | Non-classical monocytes | 2.52 | down in FLARE |
| 2 | Tcm/Naive cytotoxic T cells | 2.42 | down in FLARE |
| 3 | Naive B cells | 1.90 | down in FLARE |
| 4 | Regulatory T cells | 1.85 | down in FLARE |
| 5 | Memory B cells | 1.63 | down in FLARE |
| 6 | Type 17 helper T cells | 1.59 | down in FLARE |
| 7 | MAIT cells | 1.54 | down in FLARE |
| 8 | Tem/Effector helper T cells | 1.54 | down in FLARE |
| **9** | **NKT cells** | **1.51** | **up in FLARE** |
| 10 | Macrophages | 1.35 | down in FLARE |
| 11 | Tcm/Naive helper T cells | 1.28 | down in FLARE |
| ... | ... | ... | ... |
| 25 | Monocytes | 0.05 | down in FLARE |
| 26–35 | B cells, Megakaryocytes, etc. | 0.00 | no signal |

Key finding: TNF is **down** in flare for most T cell and monocyte subsets.
Only NKT cells have TNF up in flare among the clean cell types.
The paper reported TNF +30x in "naive CD4+ T cells" — in this finer
annotation that population maps to Tcm/Naive helper T cells, but the
t-stat there is -1.28 (down in flare). This may reflect the difference
between log-fold-change (old metric) and t-statistics (new metric), or
the cell population annotation.

## Experiment Design

### Why These 6

Combined uses all 35 clean cell types for maximum signal aggregation.
Per-cell-type experiments focus on:
- **Tcm/Naive helper T cells**: closest to "naive CD4+ T cells" from
  the paper. TNF = -1.28 (abs=1.28), 11th strongest.
- **Non-classical monocytes**: strongest TNF abs signal (2.52).
- **Classical monocytes**: comparison to Phase 5.4 "Monocytes only"
  (those were a different cell population in the broad annotation).
- **Tcm/Naive cytotoxic T cells**: 2nd strongest TNF signal (2.42).
- **NKT cells**: only clean cell type with TNF positive (+up in FLARE,
  abs=1.51). Biologically distinct direction.

### Combined Experiment Risk

Phase 5.4 combined (5 cell types, min_tstat=0.0): 32,118 composite nodes.
Phase 5.6 combined (35 cell types): could be ~100,000–200,000 composite
nodes. If training collapses (constant scores, high variance across seeds),
it will show the same pattern as Phase 5.5 combined (mean rank ~28,000,
std ~26,000). If it collapses, per-cell-type results are still valid.

## Drugs of Interest (unchanged from Phase 5.5)

| Drug | CHEMBL ID | Mechanism |
|------|-----------|-----------|
| Adalimumab | CHEMBL1201580 | Anti-TNF (held out) |
| Siltuximab | CHEMBL1743070 | Anti-IL-6 (supervision) |
| Tocilizumab | CHEMBL1237022 | Anti-IL-6R (supervision) |

## Expected Output Structure

```
results/phase_5_6/
  combined/phase_5_2_results.json
  tcm_naive_helper_t/phase_5_2_results.json
  non_classical_mono/phase_5_2_results.json
  classical_mono/phase_5_2_results.json
  tcm_naive_cytotoxic_t/phase_5_2_results.json
  nkt_cells/phase_5_2_results.json
  phase_5_6_summary.json
```

## Files

| File | Role |
|------|------|
| `data/experimental/iMCD_TAFRO_cell_specific_tstats 2.csv` | New 42-cell-type input |
| `src/enhanced_kgnn/composite_node_builder.py` | abs(tstat) change (line 275) |
| `jobs/run_phase_5_6.py` | 6-experiment entry point |
| `jobs/phase_5_6_finetune.sh` | SLURM job (48h, 64GB) |

## Status

- [x] `composite_node_builder.py` updated to use abs(tstat)
- [x] New test `test_negative_tstat_included_as_abs_value` added
- [x] `run_phase_5_6.py` written with 35-clean-cell-type explicit list
- [x] `phase_5_6_finetune.sh` written
- [ ] Tests run locally
- [ ] Commit, push, sbatch on Sockeye
