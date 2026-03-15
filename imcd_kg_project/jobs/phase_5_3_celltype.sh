#!/bin/bash
#SBATCH --job-name=phase_5_3
#SBATCH --account=st-singha53-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/phase_5_3_%j.txt

# phase_5_3_celltype.sh
# ---------------------
# Phase 5.3: Per-cell-type composite node experiments.
#
# What this does:
#   Runs 6 experiments using ALL ~10,900 matched genes (not just the
#   68 direct Castleman neighbors used in Phase 5.2):
#     1. Combined   — all cell types together (seeds=[42,123,303])
#     2. B cells    — B cells only            (seeds=[42])
#     3. ILC        — ILC only                (seeds=[42])
#     4. Megakaryocytes/platelets only         (seeds=[42])
#     5. Monocytes  — Monocytes only          (seeds=[42])
#     6. T cells    — T cells only            (seeds=[42])
#
# Professor's hypothesis:
#   T cells ranking should be highest for adalimumab because TNF
#   upregulation in naive CD4+ T cells is the key finding in the paper.
#
# Results saved to: results/phase_5_3/
#   phase_5_3_summary.json  — all 6 experiments side by side
#   combined/               — combined result (3 seeds)
#   b_cells/                — B cells result
#   ilc/                    — ILC result
#   megakaryocytes/         — Megakaryocytes result
#   monocytes/              — Monocytes result
#   t_cells/                — T cells result

module load gcc/9.4.0 miniconda3/4.9.2
source activate imcd_kg

cd /scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project

echo "========================================================"
echo "PHASE 5.3: Per-Cell-Type Composite Node Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================================"

python jobs/run_phase_5_3.py

echo "========================================================"
echo "End: $(date)"
echo "========================================================"
