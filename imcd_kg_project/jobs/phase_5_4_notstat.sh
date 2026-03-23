#!/bin/bash
#SBATCH --job-name=phase_5_4
#SBATCH --account=st-singha53-1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/phase_5_4_%j.txt

# phase_5_4_notstat.sh
# --------------------
# Phase 5.4: same 6 experiments as Phase 5.3 but with min_tstat=0.0.
# All non-zero t-stats from the CSV are included as composite nodes.
#
# Expected composite node counts (approximate):
#   Combined:      ~35,000 nodes (all cell types, all non-zero genes)
#   T cells:       ~11,700 nodes
#   Monocytes:     ~9,500  nodes
#   ILC:           ~7,200  nodes
#   B cells:       ~5,800  nodes
#   Megakaryocytes:~1,300  nodes
#
# Professor's hypothesis: with all t-stats included, T cells will have
# the most composite nodes and should produce the highest adalimumab
# ranking for Castleman disease.
#
# Results saved to: results/phase_5_4/

module load gcc/9.4.0 miniconda3/4.9.2
source activate imcd_kg

cd /scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project

echo "========================================================"
echo "PHASE 5.4: All T-Stats (min_tstat=0.0)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================================"

python jobs/run_phase_5_4.py

echo "========================================================"
echo "End: $(date)"
echo "========================================================"
