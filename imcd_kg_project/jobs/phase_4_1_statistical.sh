#!/bin/bash
#SBATCH --job-name=phase_4_1_stat
#SBATCH --account=st-singha53-1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/phase_4_1_%j.txt

module load gcc/9.4.0 miniconda3/4.9.2
source activate imcd_kg

cd /scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project

python - << 'PYEOF'
import sys
sys.path.insert(0, 'src')

from enhanced_kgnn.enhanced_predictor import ExperimentalGraphPredictor, GraphSAGEModel
import torch
import numpy as np
import json
from pathlib import Path
from scipy import stats

print("="*60)
print("PHASE 4.1: STATISTICAL VALIDATION")
print("="*60)

# Initialize predictor
predictor = ExperimentalGraphPredictor()

# 10 different random seeds
seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
trial_results = []

# Run 10 trials
for i, seed in enumerate(seeds, 1):
    print(f"\n{'='*60}")
    print(f"TRIAL {i}/10 (Seed: {seed})")
    print(f"{'='*60}")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Run comparison
    trial = predictor.compare_predictions(epochs=200)
    
    # Store results
    trial_results.append({
        'trial': i,
        'seed': seed,
        'baseline_rank': trial['baseline_adalimumab_rank'],
        'enhanced_rank': trial['enhanced_adalimumab_rank'],
        'improvement': trial['ranking_improvement'],
        'baseline_score': trial['baseline_adalimumab_score'],
        'enhanced_score': trial['enhanced_adalimumab_score']
    })
    
    print(f"\nResults:")
    print(f"  Baseline:    #{trial['baseline_adalimumab_rank']:,} (score: {trial['baseline_adalimumab_score']:.4f})")
    print(f"  Enhanced:    #{trial['enhanced_adalimumab_rank']:,} (score: {trial['enhanced_adalimumab_score']:.4f})")
    print(f"  Improvement: +{trial['ranking_improvement']:,} positions")

# Calculate statistics
print(f"\n{'='*60}")
print("STATISTICAL ANALYSIS")
print(f"{'='*60}")

baseline_ranks = [r['baseline_rank'] for r in trial_results]
enhanced_ranks = [r['enhanced_rank'] for r in trial_results]
improvements = [r['improvement'] for r in trial_results]

# Descriptive statistics
baseline_mean = np.mean(baseline_ranks)
baseline_std = np.std(baseline_ranks, ddof=1)
enhanced_mean = np.mean(enhanced_ranks)
enhanced_std = np.std(enhanced_ranks, ddof=1)
improvement_mean = np.mean(improvements)
improvement_std = np.std(improvements, ddof=1)

# Confidence intervals (95%)
from scipy.stats import t as t_dist
ci_95 = t_dist.interval(0.95, len(improvements)-1, 
                        loc=improvement_mean, 
                        scale=improvement_std/np.sqrt(len(improvements)))

# Paired t-test
t_stat, p_value = stats.ttest_rel(baseline_ranks, enhanced_ranks)

# Effect size (Cohen's d)
cohens_d = improvement_mean / improvement_std

print(f"\nBaseline Rankings:")
print(f"  Mean: {baseline_mean:,.0f}")
print(f"  Std:  {baseline_std:,.0f}")
print(f"  Range: [{min(baseline_ranks):,}, {max(baseline_ranks):,}]")

print(f"\nEnhanced Rankings:")
print(f"  Mean: {enhanced_mean:,.0f}")
print(f"  Std:  {enhanced_std:,.0f}")
print(f"  Range: [{min(enhanced_ranks):,}, {max(enhanced_ranks):,}]")

print(f"\nImprovement:")
print(f"  Mean: {improvement_mean:,.0f} ± {improvement_std:,.0f}")
print(f"  95% CI: [{ci_95[0]:,.0f}, {ci_95[1]:,.0f}]")
print(f"  Range: [{min(improvements):,}, {max(improvements):,}]")

print(f"\nStatistical Tests:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Cohen's d: {cohens_d:.4f}")

if p_value < 0.001:
    print(f"  ✅ HIGHLY SIGNIFICANT (p < 0.001)")
elif p_value < 0.01:
    print(f"  ✅ VERY SIGNIFICANT (p < 0.01)")
elif p_value < 0.05:
    print(f"  ✅ SIGNIFICANT (p < 0.05)")
else:
    print(f"  ❌ NOT SIGNIFICANT (p >= 0.05)")
    print(f"  ⚠️  WARNING: Improvement may not be statistically significant!")

# Save results
results_dir = Path('results/phase_4_1_statistical')
results_dir.mkdir(parents=True, exist_ok=True)

# Save trial details
with open(results_dir / 'trial_results.json', 'w') as f:
    json.dump(trial_results, f, indent=2)

# Save summary statistics
summary = {
    'num_trials': len(seeds),
    'seeds': seeds,
    'baseline': {
        'mean': float(baseline_mean),
        'std': float(baseline_std),
        'min': int(min(baseline_ranks)),
        'max': int(max(baseline_ranks))
    },
    'enhanced': {
        'mean': float(enhanced_mean),
        'std': float(enhanced_std),
        'min': int(min(enhanced_ranks)),
        'max': int(max(enhanced_ranks))
    },
    'improvement': {
        'mean': float(improvement_mean),
        'std': float(improvement_std),
        'min': int(min(improvements)),
        'max': int(max(improvements)),
        'ci_95_lower': float(ci_95[0]),
        'ci_95_upper': float(ci_95[1])
    },
    'statistics': {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05)
    }
}

with open(results_dir / 'statistical_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("✅ PHASE 4.1 COMPLETE")
print(f"{'='*60}")
print(f"\nResults saved to: {results_dir}")
print(f"  - trial_results.json (all 10 trials)")
print(f"  - statistical_summary.json (summary statistics)")

PYEOF