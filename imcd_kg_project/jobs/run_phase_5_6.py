"""
run_phase_5_6.py
----------------
Entry point for Phase 5.6: finer cell type resolution using the new
42-cell-type CSV (iMCD_TAFRO_cell_specific_tstats 2.csv) with abs(t-stat).

Changes from Phase 5.5:
  1. New CSV: iMCD_TAFRO_cell_specific_tstats 2.csv (42 cell types,
     t-statistics with remission as reference, + means up in FLARE).
  2. abs(t-stat) instead of tstat > 0 filter — includes all differentially
     expressed genes regardless of direction (per Prof. Singh's question).
  3. 7 cell types excluded due to extreme/broken t-stat values:
     Alveolar macrophages, CD8a/a, Double-negative thymocytes,
     Double-positive thymocytes, Memory CD4+ cytotoxic T cells,
     T(agonist), Transitional NK.
  4. 35 clean cell types remain. The combined experiment passes them
     explicitly so the 7 broken columns are never loaded.
  5. Per-cell-type experiments focus on the most biologically relevant
     subsets based on TNF abs(t-stat) ranking and proximity to the
     naive CD4+ T cell signal described in the paper.

Experiment design:
  combined                — all 35 clean cell types, seeds=[42, 123, 303]
  tcm_naive_helper_t      — Tcm/Naive helper T cells (closest to naive CD4+
                            T cells from paper), seed=[42]
  non_classical_mono      — Non-classical monocytes (strongest TNF signal,
                            abs=2.52), seed=[42]
  classical_mono          — Classical monocytes (direct comparison with
                            Phase 5.4 "Monocytes only"), seed=[42]
  tcm_naive_cytotoxic_t   — Tcm/Naive cytotoxic T cells (2nd strongest
                            TNF abs signal, abs=2.42), seed=[42]
  nkt_cells               — NKT cells (only clean cell type with positive
                            TNF t-stat > 1.0, i.e. up in FLARE), seed=[42]

Results saved to:
  results/phase_5_6/combined/phase_5_2_results.json
  results/phase_5_6/tcm_naive_helper_t/phase_5_2_results.json
  results/phase_5_6/non_classical_mono/phase_5_2_results.json
  results/phase_5_6/classical_mono/phase_5_2_results.json
  results/phase_5_6/tcm_naive_cytotoxic_t/phase_5_2_results.json
  results/phase_5_6/nkt_cells/phase_5_2_results.json
  results/phase_5_6/phase_5_6_summary.json

Run via SLURM:
  sbatch jobs/phase_5_6_finetune.sh
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
CSV_PATH    = Path("data/experimental/iMCD_TAFRO_cell_specific_tstats 2.csv")
RESULTS_DIR = Path("results/phase_5_6")
EPOCHS      = 200
MIN_TSTAT   = 0.0

# 7 cell types with extreme/broken t-stat values (up to ~8.5 million).
# Confirmed broken in both file 1 and file 2. Never load these columns.
EXCLUDED_CELL_TYPES = {
    "Alveolar macrophages",
    "CD8a/a",
    "Double-negative thymocytes",
    "Double-positive thymocytes",
    "Memory CD4+ cytotoxic T cells",
    "T(agonist)",
    "Transitional NK",
}

# All 35 clean cell types from the 42-column CSV.
# Passed explicitly to combined experiment so excluded columns are never read.
ALL_35_CELL_TYPES = [
    "Age-associated B cells",
    "B cells",
    "CD16- NK cells",
    "CD16+ NK cells",
    "CD8a/b(entry)",
    "Classical monocytes",
    "CRTAM+ gamma-delta T cells",
    "DC1",
    "DC2",
    "Follicular helper T cells",
    "HSC/MPP",
    "ILC3",
    "Macrophages",
    "MAIT cells",
    "Megakaryocytes/platelets",
    "Memory B cells",
    "MNP",
    "Monocytes",
    "Myelocytes",
    "Naive B cells",
    "NK cells",
    "NKT cells",
    "Non-classical monocytes",
    "pDC",
    "Plasma cells",
    "Regulatory T cells",
    "Tcm/Naive cytotoxic T cells",
    "Tcm/Naive helper T cells",
    "Tem/Effector helper T cells",
    "Tem/Temra cytotoxic T cells",
    "Tem/Trm cytotoxic T cells",
    "Transitional B cells",
    "Treg(diff)",
    "Trm cytotoxic T cells",
    "Type 17 helper T cells",
]

assert len(ALL_35_CELL_TYPES) == 35, f"Expected 35 clean cell types, got {len(ALL_35_CELL_TYPES)}"

EXPERIMENTS = [
    {
        "name":       "combined",
        "label":      "All 35 clean cell types combined",
        "cell_types": ALL_35_CELL_TYPES,
        "seeds":      [42, 123, 303],
    },
    {
        "name":       "tcm_naive_helper_t",
        "label":      "Tcm/Naive helper T cells only",
        "cell_types": ["Tcm/Naive helper T cells"],
        "seeds":      [42],
    },
    {
        "name":       "non_classical_mono",
        "label":      "Non-classical monocytes only",
        "cell_types": ["Non-classical monocytes"],
        "seeds":      [42],
    },
    {
        "name":       "classical_mono",
        "label":      "Classical monocytes only",
        "cell_types": ["Classical monocytes"],
        "seeds":      [42],
    },
    {
        "name":       "tcm_naive_cytotoxic_t",
        "label":      "Tcm/Naive cytotoxic T cells only",
        "cell_types": ["Tcm/Naive cytotoxic T cells"],
        "seeds":      [42],
    },
    {
        "name":       "nkt_cells",
        "label":      "NKT cells only",
        "cell_types": ["NKT cells"],
        "seeds":      [42],
    },
]

print("=" * 60)
print("PHASE 5.6: Finer cell type resolution (42 cell types, abs t-stat)")
print("=" * 60)
print(f"  Graph:      {GRAPH_PATH}")
print(f"  CSV:        {CSV_PATH}")
print(f"  Epochs:     {EPOCHS}")
print(f"  Min t-stat: {MIN_TSTAT}  (all non-zero abs t-stats included)")
print(f"  Results:    {RESULTS_DIR}")
print(f"  Experiments: {len(EXPERIMENTS)}")
print(f"  Excluded cell types ({len(EXCLUDED_CELL_TYPES)}): {sorted(EXCLUDED_CELL_TYPES)}")
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
        "drug_mean_ranks_castleman": {
            k: {kk: float(vv) if hasattr(vv, 'item') else vv for kk, vv in v.items()}
            for k, v in (result.drug_mean_ranks_castleman or {}).items()
        },
    }

    print(result.summary())

# Save cross-experiment summary
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
summary_path = RESULTS_DIR / "phase_5_6_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 60)
print("PHASE 5.6 COMPLETE — Cross-experiment summary")
print("=" * 60)
for name, s in summary.items():
    rank = s["castleman_mean_rank"]
    rank_str = f"#{rank:,.0f}" if rank else "N/A"
    spec = "SPECIFIC" if s["disease_specific"] else "NOT SPECIFIC"
    print(f"  {s['label']:45s}  Castleman={rank_str:>10}  {spec}")
print()
print(f"Full summary saved to: {summary_path}")
