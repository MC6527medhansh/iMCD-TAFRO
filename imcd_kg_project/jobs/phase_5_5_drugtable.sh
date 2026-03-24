#!/bin/bash
#SBATCH --job-name=phase_5_5
#SBATCH --account=st-singha53-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/phase_5_5_%j.txt

# phase_5_5_drugtable.sh
# ----------------------
# Phase 5.5: same 6 experiments as Phase 5.4 (min_tstat=0.0) but now
# saves the top-200 drug ranking table (rank, name, mechanism, score)
# per seed per experiment.
#
# Professor Singh asked to see how known Castleman drugs (siltuximab,
# tocilizumab, adalimumab, etanercept, rituximab, corticosteroids) rank
# across the combined and per-cell-type experiments.
#
# Results saved to: results/phase_5_5/

module load gcc/9.4.0 miniconda3/4.9.2
source activate imcd_kg

cd /scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project

echo "========================================================"
echo "PHASE 5.5: Drug Ranking Table (min_tstat=0.0)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================================"

python jobs/run_phase_5_5.py

echo "========================================================"
echo "End: $(date)"
echo "========================================================"
