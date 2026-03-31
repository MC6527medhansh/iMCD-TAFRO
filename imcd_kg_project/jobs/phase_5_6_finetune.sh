#!/bin/bash
#SBATCH --job-name=phase_5_6
#SBATCH --account=st-singha53-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/phase_5_6_%j.txt

# phase_5_6_finetune.sh
# ---------------------
# Phase 5.6: finer cell type resolution using 42-cell-type CSV (file 2).
# New features vs Phase 5.5:
#   - abs(t-stat) filter: includes all differentially expressed genes
#   - 7 broken cell types excluded (extreme values up to ~8.5M)
#   - 35 clean cell types
#   - 6 experiments: combined + 5 biologically motivated cell types
#
# Combined experiment risk: with 35 cell types and abs(t-stat), composite
# node count may exceed Phase 5.4 (32,118). If training collapses on the
# combined, per-cell-type results are still valid.
#
# Results saved to: results/phase_5_6/

module load gcc/9.4.0 miniconda3/4.9.2
source activate imcd_kg

cd /scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project

echo "========================================================"
echo "PHASE 5.6: Finer cell type resolution (abs t-stat)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================================"

python jobs/run_phase_5_6.py

echo "========================================================"
echo "End: $(date)"
echo "========================================================"
