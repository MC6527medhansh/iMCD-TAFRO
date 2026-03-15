"""
run_phase_5_3.py
----------------
Entry point for Phase 5.3 cell-type comparison job on Sockeye.

What this does:
  Runs 6 fine-tuning experiments using the updated composite node
  architecture (all ~10,900 matched genes, not just Castleman's 68
  direct neighbors):

    1. Combined   — all five cell types together  (seeds=[42, 123, 303])
    2. B cells    — B cells only                  (seeds=[42])
    3. ILC        — ILC only                      (seeds=[42])
    4. Megakaryocytes/platelets only               (seeds=[42])
    5. Monocytes  — Monocytes only                (seeds=[42])
    6. T cells    — T cells only                  (seeds=[42])

  For each experiment, adalimumab is evaluated for:
    - Castleman disease
    - Type 2 Diabetes, Hypertension, Alzheimer (non-TNF controls)

  The professor's hypothesis: T cells should give the highest ranking
  for adalimumab because TNF upregulation in naive CD4+ T cells is the
  key finding in the paper. Monocytes may also rank well (t-stat=32.22).

Results saved to:
  results/phase_5_3/combined/phase_5_2_results.json
  results/phase_5_3/b_cells/phase_5_2_results.json
  results/phase_5_3/ilc/phase_5_2_results.json
  results/phase_5_3/megakaryocytes/phase_5_2_results.json
  results/phase_5_3/monocytes/phase_5_2_results.json
  results/phase_5_3/t_cells/phase_5_2_results.json

Run via SLURM:
  sbatch jobs/phase_5_3_celltype.sh

Or directly:
  conda run -n imcd_kg python jobs/run_phase_5_3.py
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
RESULTS_DIR = Path("results/phase_5_3")
EPOCHS      = 200
MIN_TSTAT   = 2.0

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
print("PHASE 5.3: Per-Cell-Type Composite Node Experiments")
print("=" * 60)
print(f"  Graph:      {GRAPH_PATH}")
print(f"  CSV:        {CSV_PATH}")
print(f"  Epochs:     {EPOCHS}")
print(f"  Min t-stat: {MIN_TSTAT}")
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
        "label":              exp["label"],
        "cell_types":         exp["cell_types"],
        "seeds":              exp["seeds"],
        "castleman_mean_rank": result.castleman_mean_rank,
        "castleman_std_rank":  result.castleman_std_rank,
        "non_tnf_mean_ranks":  result.non_tnf_mean_ranks,
        "disease_specific":    result.disease_specific,
    }

    print(result.summary())

# Save cross-experiment summary
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary_path = RESULTS_DIR / "phase_5_3_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 60)
print("PHASE 5.3 COMPLETE — Cross-experiment summary")
print("=" * 60)
for name, s in summary.items():
    rank = s["castleman_mean_rank"]
    rank_str = f"#{rank:,.0f}" if rank else "N/A"
    spec = "SPECIFIC" if s["disease_specific"] else "NOT SPECIFIC"
    print(f"  {s['label']:35s}  Castleman={rank_str:>10}  {spec}")
print()
print(f"Full summary saved to: {summary_path}")
