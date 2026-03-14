#!/bin/bash
#SBATCH --job-name=phase_5_2
#SBATCH --account=st-singha53-1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/phase_5_2_%j.txt

# phase_5_2_finetune.sh
# ---------------------
# Phase 5.2: Train GATConv on composite-node-augmented graph.
#
# What this does:
#   1. Loads RTX-KG2 full_graph.pkl
#   2. Adds composite (gene, cell_type) intermediate nodes using t-stats
#      from iMCD_TAFRO_cell_specific_tstats.csv (min_tstat=2.0)
#   3. Trains GATConv with siltuximab + tocilizumab as Castleman supervision
#   4. Evaluates whether adalimumab emerges as disease-specific for Castleman
#
# Results saved to: results/phase_5_2/phase_5_2_results.json
#
# Success criterion:
#   Adalimumab should rank better (lower number) for Castleman than
#   for non-TNF diseases (diabetes, hypertension, Alzheimer).

module load gcc/9.4.0 miniconda3/4.9.2

cd /scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project

echo "========================================================"
echo "PHASE 5.2: Composite Node Fine-Tuning"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================================"

conda run -n imcd_kg python jobs/run_phase_5_2.py

echo "========================================================"
echo "End: $(date)"
echo "========================================================"
