"""
run_phase_5_5.py
----------------
Entry point for Phase 5.5: same 6 experiments as Phase 5.4 (min_tstat=0.0),
extended to save the top-200 drug ranking table per seed per experiment.

Why:
  Phases 5.2-5.4 only tracked one number: adalimumab's rank for Castleman.
  Professor Singh asked to see the full ranked drug list — rank, drug name,
  mechanism, score — so we can answer: are known Castleman drugs rising to
  the top? where does adalimumab sit relative to other TNF inhibitors? do
  corticosteroids cluster together?

  This is the same experimental setup as Phase 5.4. No parameters change.
  We re-run purely because model weights are not saved between phases.

What is saved per experiment:
  results/phase_5_5/<experiment>/phase_5_2_results.json
    - castleman_mean_rank, castleman_std_rank (adalimumab, held out)
    - non_tnf_mean_ranks (adalimumab for diabetes/hypertension/Alzheimer)
    - disease_specific verdict
    - n_composite_nodes, min_tstat, cell_types, seeds, epochs
    - seed_results: per-seed adalimumab rank + top_drugs list (top 200)
      top_drugs format: [{rank, drug_id, name, mechanism, score}, ...]

  results/phase_5_5/phase_5_5_summary.json
    - cross-experiment summary (adalimumab rank + verdict for all 6)

Note on top_drugs:
  Siltuximab and tocilizumab will always appear at the top (rank 1-2) because
  they are positive supervision pairs during training. This is expected and
  not informative. The interesting signal is in the ordering of drugs that
  were NOT in supervision: adalimumab, etanercept, rituximab, etc.

  Adalimumab itself ranks at ~#12,000 and will NOT appear in the top-200 table.
  Its rank is tracked separately in castleman_mean_rank.

Run via SLURM:
  sbatch jobs/phase_5_5_drugtable.sh
"""
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

from enhanced_kgnn.finetuner import CompositeFinetuner

GRAPH_PATH  = Path("data/processed/full_graph.pkl")
CSV_PATH    = Path("data/experimental/iMCD_TAFRO_cell_specific_tstats.csv")
RESULTS_DIR = Path("results/phase_5_5")
EPOCHS      = 200
MIN_TSTAT   = 0.0

EXPERIMENTS = [
    {
        "name":       "combined",
        "label":      "All cell types combined",
        "cell_types": None,
        "seeds":      [42, 123, 303],
    },
    {
        "name":       "b_cells",
        "label":      "B cells only",
        "cell_types": ["B cells"],
        "seeds":      [42],
    },
    {
        "name":       "ilc",
        "label":      "ILC only",
        "cell_types": ["ILC"],
        "seeds":      [42],
    },
    {
        "name":       "megakaryocytes",
        "label":      "Megakaryocytes/platelets only",
        "cell_types": ["Megakaryocytes/platelets"],
        "seeds":      [42],
    },
    {
        "name":       "monocytes",
        "label":      "Monocytes only",
        "cell_types": ["Monocytes"],
        "seeds":      [42],
    },
    {
        "name":       "t_cells",
        "label":      "T cells only",
        "cell_types": ["T cells"],
        "seeds":      [42],
    },
]

print("=" * 60)
print("PHASE 5.5: Drug Ranking Table (min_tstat=0.0)")
print("=" * 60)
print(f"  Graph:      {GRAPH_PATH}")
print(f"  CSV:        {CSV_PATH}")
print(f"  Epochs:     {EPOCHS}")
print(f"  Min t-stat: {MIN_TSTAT}  (all non-zero t-stats included)")
print(f"  Results:    {RESULTS_DIR}")
print(f"  Experiments: {len(EXPERIMENTS)}")
print()

summary = {}

for exp in EXPERIMENTS:
    print()
    print(f"{'='*60}")
    print(f"Experiment: {exp['label']}")
    print(f"{'='*60}")

    finetuner = CompositeFinetuner(min_tstat=MIN_TSTAT)
    result = finetuner.run(
        graph_path=GRAPH_PATH,
        csv_path=CSV_PATH,
        seeds=exp["seeds"],
        epochs=EPOCHS,
        results_dir=RESULTS_DIR / exp["name"],
        cell_types=exp["cell_types"],
    )

    summary[exp["name"]] = {
        "label":               exp["label"],
        "cell_types":          exp["cell_types"],
        "seeds":               exp["seeds"],
        "castleman_mean_rank": result.castleman_mean_rank,
        "castleman_std_rank":  result.castleman_std_rank,
        "non_tnf_mean_ranks":  result.non_tnf_mean_ranks,
        "disease_specific":    bool(result.disease_specific) if result.disease_specific is not None else None,
    }

    print(result.summary())

# Save cross-experiment summary
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary_path = RESULTS_DIR / "phase_5_5_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 60)
print("PHASE 5.5 COMPLETE — Cross-experiment summary")
print("=" * 60)
for name, s in summary.items():
    rank = s["castleman_mean_rank"]
    rank_str = f"#{rank:,.0f}" if rank else "N/A"
    spec = "SPECIFIC" if s["disease_specific"] else "NOT SPECIFIC"
    print(f"  {s['label']:35s}  Castleman={rank_str:>10}  {spec}")
print()
print(f"Full summary saved to: {summary_path}")
