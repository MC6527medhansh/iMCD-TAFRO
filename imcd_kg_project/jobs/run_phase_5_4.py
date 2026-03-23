"""
run_phase_5_4.py
----------------
Entry point for Phase 5.4: same 6 experiments as Phase 5.3 but with
min_tstat=0.0, meaning all non-zero t-stats are included.

Why:
  Phase 5.3 used min_tstat=2.0. This cut off a large number of genes
  with low but non-zero t-stats. The per-cell-type experiments produced
  similar node counts across cell types (949 to 2,409), which may have
  contributed to the similar and noisy per-cell-type rankings.

  With min_tstat=0.0, all non-zero t-stats are included:
    T cells:             ~11,700 composite nodes
    Monocytes:           ~9,500 composite nodes
    ILC:                 ~7,200 composite nodes
    B cells:             ~5,800 composite nodes
    Megakaryocytes:      ~1,300 composite nodes

  T cells now has significantly more composite nodes than monocytes.
  The professor's hypothesis is that T cells should produce the highest
  ranking for adalimumab. With more nodes and denser coverage, the T
  cell signal may become stronger and clearer.

  Exact zeros in the CSV are excluded regardless of this threshold
  because _load_tstats_all_genes filters tstat > 0 before min_tstat
  is applied.

Results saved to:
  results/phase_5_4/combined/phase_5_2_results.json
  results/phase_5_4/b_cells/phase_5_2_results.json
  results/phase_5_4/ilc/phase_5_2_results.json
  results/phase_5_4/megakaryocytes/phase_5_2_results.json
  results/phase_5_4/monocytes/phase_5_2_results.json
  results/phase_5_4/t_cells/phase_5_2_results.json
  results/phase_5_4/phase_5_4_summary.json

Run via SLURM:
  sbatch jobs/phase_5_4_notstat.sh
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
RESULTS_DIR = Path("results/phase_5_4")
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
print("PHASE 5.4: All T-Stats (min_tstat=0.0)")
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
summary_path = RESULTS_DIR / "phase_5_4_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 60)
print("PHASE 5.4 COMPLETE — Cross-experiment summary")
print("=" * 60)
for name, s in summary.items():
    rank = s["castleman_mean_rank"]
    rank_str = f"#{rank:,.0f}" if rank else "N/A"
    spec = "SPECIFIC" if s["disease_specific"] else "NOT SPECIFIC"
    print(f"  {s['label']:35s}  Castleman={rank_str:>10}  {spec}")
print()
print(f"Full summary saved to: {summary_path}")
