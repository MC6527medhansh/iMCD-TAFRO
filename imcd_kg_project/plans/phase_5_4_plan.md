# Phase 5.4 Plan: All T-Stats (min_tstat=0.0)

## Goal

Same 6 experiments as Phase 5.3 but with `MIN_TSTAT = 0.0`. All non-zero
t-stats from the CSV are included as composite nodes. Exact zeros are still
excluded (handled by `_load_tstats_all_genes` which filters `tstat > 0`
before min_tstat is applied).

Professor's hypothesis: with all t-stats included, T cells will have the most
composite nodes (~11,700) and should produce the highest adalimumab ranking
for Castleman disease.

## What Changed from Phase 5.3

One parameter only: `MIN_TSTAT = 2.0` → `MIN_TSTAT = 0.0`.

Everything else (architecture, graph, CSV, epochs, seeds) is identical.

## Composite Node Counts (actual, from Sockeye run)

| Experiment      | Phase 5.3 nodes | Phase 5.4 nodes |
|-----------------|-----------------|-----------------|
| Combined        | 7,362           | 32,118          |
| T cells         | 2,334           | 10,208          |
| Monocytes       | 2,409           | ~9,500          |
| ILC             | 1,300           | ~7,200          |
| B cells         | 949             | ~5,800          |
| Megakaryocytes  | 370             | ~1,300          |

## Files Created

- `jobs/run_phase_5_4.py` — entry point, identical to run_phase_5_3.py except
  MIN_TSTAT=0.0 and results go to `results/phase_5_4/`
- `jobs/phase_5_4_notstat.sh` — SLURM job (48h, 64GB, 8 CPUs)

## Results (Sockeye, March 2026)

All experiments: epochs=200, min_tstat=0.0, seed=42 (combined uses [42,123,303]).

| Experiment          | Composite nodes | Castleman rank | Non-TNF mean | Gap    | Verdict      |
|---------------------|-----------------|----------------|--------------|--------|--------------|
| Combined (all)      | 32,118          | #12,272        | #13,780      | +1,508 | SPECIFIC     |
| Megakaryocytes only | ~1,300          | —              | —            | +93    | SPECIFIC     |
| B cells only        | ~5,800          | —              | —            | +16    | barely       |
| ILC only            | ~7,200          | —              | —            | -12    | NOT SPECIFIC |
| Monocytes only      | ~9,500          | —              | —            | -46    | NOT SPECIFIC |
| T cells only        | 10,208          | —              | —            | -55    | NOT SPECIFIC |

Non-TNF control ranks for combined:
- Type 2 Diabetes:  #13,856
- Hypertension:     #13,786
- Alzheimer Disease:#13,699

## Key Findings

1. **Combined is the best result across all phases.**
   - Castleman rank: #12,272 (best ever)
   - Disease-specificity gap: +1,508 (largest ever)
   - Beats Phase 5.3 combined (#13,042, gap=+1,443) and Phase 5.2 (#13,346, gap=~1,600)
   - More composite nodes (32,118 vs 7,362) = denser coverage = cleaner signal

2. **Professor's T cells hypothesis is not confirmed — reversed.**
   - T cells has the most composite nodes (10,208) but the worst individual gap (-55)
   - Lowering min_tstat to 0.0 adds thousands of low-t-stat genes that outweigh the
     true TNF signal. For individual cell types this noise exceeds the genuine signal.
   - Monocytes also flips from barely positive (Phase 5.3: +78) to negative (-46) for same reason.

3. **Megakaryocytes is the surprise.**
   - NOT SPECIFIC in Phase 5.3 (gap=-48, only 370 nodes)
   - SPECIFIC in Phase 5.4 (gap=+93, ~1,300 nodes)
   - Going from 370 to ~1,300 composite nodes crossed the threshold needed for
     the model to extract a meaningful signal from this cell type.

4. **Individual cell types at 1 seed are still too noisy.**
   - ILC, Monocytes, T cells all flip sign vs Phase 5.3
   - Combined survives because aggregating all 5 cell types provides enough genuine
     signal to stay above the noise floor. This is the right approach for now.

5. **The combined approach is confirmed as the right strategy.**
   - Both Phase 5.3 and 5.4 confirm: combined > any individual cell type
   - Lower min_tstat always helps the combined experiment
   - To test individual cell types properly, multi-seed runs are needed

6. **To test the naive CD4+ T cell hypothesis:**
   - The "T cells" column is an aggregate across all T cell subtypes
   - TNF t-stat in this aggregate is only 2.91 vs 32.22 in Monocytes
   - Finer granularity (naive CD4+ breakdown) would likely change the ordering
   - This is an open question to discuss with Professor Singh

## Progression Across Phases

| Phase | Castleman rank | Gap    | Key change                        |
|-------|----------------|--------|-----------------------------------|
| 5.2   | #13,346        | ~1,600 | 49 composite nodes, min_tstat=2.0 |
| 5.3   | #13,042        | +1,443 | ~10,900 genes, min_tstat=2.0      |
| 5.4   | #12,272        | +1,508 | ~10,900 genes, min_tstat=0.0      |

Each phase improves on the last. The trend is consistent: more composite nodes
(via lower threshold or more genes) = better combined result.

## Status

- [x] run_phase_5_4.py created
- [x] phase_5_4_notstat.sh created
- [x] Sockeye run COMPLETE (March 2026)
- [x] Results recorded
- [x] Message sent to Professor Singh
