"""
run_phase_5_2.py
----------------
Entry point for Phase 5.2 fine-tuning job on Sockeye.

Run via SLURM:
  sbatch jobs/phase_5_2_finetune.sh

Or directly:
  conda run -n imcd_kg python jobs/run_phase_5_2.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

from enhanced_kgnn.finetuner import CompositeFinetuner

GRAPH_PATH   = Path("data/processed/full_graph.pkl")
CSV_PATH     = Path("data/experimental/iMCD_TAFRO_cell_specific_tstats.csv")
RESULTS_DIR  = Path("results/phase_5_2")
SEEDS        = [42, 123, 303]
EPOCHS       = 200
MIN_TSTAT    = 2.0

print("=" * 60)
print("PHASE 5.2: Composite Node Fine-Tuning")
print("=" * 60)
print(f"  Graph:      {GRAPH_PATH}")
print(f"  CSV:        {CSV_PATH}")
print(f"  Seeds:      {SEEDS}")
print(f"  Epochs:     {EPOCHS}")
print(f"  Min t-stat: {MIN_TSTAT}")
print(f"  Results:    {RESULTS_DIR}")
print()

finetuner = CompositeFinetuner(min_tstat=MIN_TSTAT)
result = finetuner.run(
    graph_path=GRAPH_PATH,
    csv_path=CSV_PATH,
    seeds=SEEDS,
    epochs=EPOCHS,
    results_dir=RESULTS_DIR,
)

print()
print(result.summary())
