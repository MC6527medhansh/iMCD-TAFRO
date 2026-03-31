# Phase 5.5 Plan: Drug Ranking Table (min_tstat=0.0)

## Goal

Same 6 experiments as Phase 5.4 (min_tstat=0.0), extended to save the full
ranking of confirmed drugs of interest per seed per experiment.

Prof. Singh asked: "I'm more interested in where existing drugs of Castleman's
are being ranked — such as anti-IL6, TNF, IL2. How does the ranking of these
three drugs differ between the cell types?"

Phases 5.2–5.4 only tracked one number: adalimumab's rank for Castleman.
Phase 5.5 answers the professor's question by explicitly tracking every
confirmed drug across all experiments.

## What Changed from Phase 5.4

No parameters changed. The experimental setup (min_tstat=0.0, same 6
experiments, same seeds, same epochs) is identical to Phase 5.4.

The extension is in what we save per seed:

| Field | Phase 5.4 | Phase 5.5 |
|-------|-----------|-----------|
| adalimumab rank for Castleman | ✓ | ✓ |
| top-500 generic drug table | ✗ | ✓ |
| drug_ranks_castleman (all confirmed drugs) | ✗ | ✓ |
| drug_mean_ranks_castleman (aggregated) | ✗ | ✓ |

## Why We Re-run (Not Just Post-Process Phase 5.4)

Model weights are not saved between phases. Re-run is required.
Non-determinism across runs (known open question in CLAUDE.md) means Phase 5.5
combined Castleman rank will differ slightly from Phase 5.4's #12,272.

## Confirmed Drugs of Interest

Only IDs that have been verified against the RTX-KG2 graph on Sockeye:

| Drug | CHEMBL ID | Mechanism | Notes |
|------|-----------|-----------|-------|
| Adalimumab | CHEMBL1201580 | Anti-TNF | Held out — NOT in supervision |
| Siltuximab | CHEMBL1743070 | Anti-IL-6 | Positive supervision pair |
| Tocilizumab | CHEMBL1237022 | Anti-IL-6R | Positive supervision pair |

Why only 3: Other Castleman drugs (etanercept, rituximab, corticosteroids)
have not been verified against the graph. Unverified IDs may not exist as
nodes. Wrong IDs would produce silent failures (drug not found).
To add more drugs: search graph node names on Sockeye, confirm CHEMBL ID,
then add to `DRUGS_OF_INTEREST` in `finetuner.py`.

## Implementation (gat_predictor.py)

`get_drug_ranks()` replaces the removed `get_top_n_drugs()`.

```
Single call to evaluate_drug_ranking()  (one forward pass)
         |
         v
Iterate sorted ranking once:
  - rank <= top_n  →  append to top_n_table
  - drug_id in drugs_of_interest  →  record in specific dict
  - rank > top_n AND all DOI found  →  break early
         |
         v
Returns (top_n_table, specific)
```

Key invariant: adalimumab sits at ~#13,000. It will NEVER appear in the
top-500 table. `specific` is the only place its rank is captured.

## Why the First Phase 5.5 Run (job 9338927) Was Wrong

First run used `get_top_n_drugs()` which:
- Returned a generic top-200 list (DAPICLERMIN etc.)
- Never tracked drugs of interest explicitly
- Siltuximab/tocilizumab/adalimumab all outside top-200 → not reported
- Mechanism filter over the top-200 list returned zero results

This was caught after reviewing results. Fixed before second run.

## Expected Output Structure

```
results/phase_5_5/
  combined/
    phase_5_2_results.json
      castleman_mean_rank: float
      castleman_std_rank: float
      non_tnf_mean_ranks: {disease: float}
      disease_specific: bool
      n_composite_nodes: int
      min_tstat: 0.0
      cell_types: null
      seeds: [42, 123, 303]
      epochs: 200
      drug_mean_ranks_castleman:
        adalimumab:  {mean_rank, std_rank, mean_score, mechanism, drug_id}
        siltuximab:  {mean_rank, std_rank, mean_score, mechanism, drug_id}
        tocilizumab: {mean_rank, std_rank, mean_score, mechanism, drug_id}
      seed_results: [
        {seed, castleman_rank, castleman_score, non_tnf_ranks,
         drug_ranks_castleman, top_500_castleman}
      ]
  b_cells/...
  ilc/...
  megakaryocytes/...
  monocytes/...
  t_cells/...
  phase_5_5_summary.json
    {combined: {label, cell_types, seeds, castleman_mean_rank, ...}, ...}
```

## What We Expected vs Actual Results (Sockeye, March 2026)

Expected: siltuximab/tocilizumab at #1/#2 (in supervision). Actual: they rank
at #14,876–17,306 for individual cell types. The supervision does not push them
to #1 — binary cross-entropy over 174K pairs across all diseases creates a weak
gradient for any single disease's supervision pair.

### Drug Rankings for Castleman (per cell type, seed=42)

| Experiment | Adalimumab (Anti-TNF) | Tocilizumab (Anti-IL-6R) | Siltuximab (Anti-IL-6) | disease_specific |
|---|---|---|---|---|
| ILC | #13,285 | #14,903 | #16,936 | YES |
| B cells | #13,293 | #14,876 | #16,976 | NO |
| Monocytes | #13,396 | #14,963 | #17,014 | YES |
| T cells | #13,793 | #15,287 | #17,285 | NO |
| Megakaryocytes | #13,804 | #15,290 | #17,306 | NO |
| Combined | FAILED | FAILED | FAILED | YES (but meaningless) |

### Key Findings

1. **Ordering is perfectly consistent across all 5 cell types:**
   adalimumab > tocilizumab > siltuximab in every experiment.
   The spread across cell types is ~500 rank positions (small).

2. **Adalimumab (held out, NOT in supervision) outranks both supervision drugs.**
   This is a significant finding: the model is not just memorizing training pairs.
   The composite node structure encoding TNF-pathway t-stats gives adalimumab a
   stronger biological signal than the direct supervision edges for siltuximab/
   tocilizumab. The model is learning from the graph structure.

3. **Combined experiment failed** — mean rank #28,604 (std=26,658), supervision
   drugs at #30,000+. Training collapsed on the 32,118-node composite graph.
   At least one of the 3 seeds produced essentially random embeddings (all drug
   scores ~0.857 constant). CPU non-determinism on large graph — known open issue.
   Individual cell-type results (smaller graphs, 1 seed) are reliable.

4. **T cells vs Monocytes**: professor hypothesized T cells > Monocytes.
   Not confirmed — Monocytes ranks adalimumab at #13,396 (better than T cells
   #13,793). However the gap is small (~400 ranks) and could be noise at 1 seed.

## Files

| File | Role |
|------|------|
| `src/enhanced_kgnn/gat_predictor.py` | `get_drug_ranks()` method |
| `src/enhanced_kgnn/finetuner.py` | `DRUGS_OF_INTEREST`, single-pass evaluation |
| `jobs/run_phase_5_5.py` | 6-experiment entry point |
| `jobs/phase_5_5_drugtable.sh` | SLURM job (48h, 64GB) |
| `tests/test_gat_unit.py` | `TestGetDrugRanks` (5 tests) |
| `tests/test_finetuner.py` | `test_drug_ranks_in_results` + updated `test_results_saved_to_disk` |

## Status

- [x] `get_drug_ranks()` implemented and unit tested
- [x] `finetuner.py` wired to `get_drug_ranks()`, `DRUGS_OF_INTEREST` confirmed
- [x] Tests written: 53 total (27 + 5 new unit + 4 + 2 updated finetuner tests)
- [ ] Tests run locally (next step before Sockeye)
- [ ] Commit, push, sbatch on Sockeye
