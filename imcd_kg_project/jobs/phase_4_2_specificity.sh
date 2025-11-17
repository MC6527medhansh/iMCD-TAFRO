#!/bin/bash
#SBATCH --job-name=phase_4_2_spec
#SBATCH --account=st-singha53-1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/phase_4_2_specificity_%j.txt

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

print("="*70)
print("PHASE 4.2: DISEASE SPECIFICITY TESTING")
print("Testing if TNF feature helps specifically for TNF-mediated diseases")
print("="*70)

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

predictor = ExperimentalGraphPredictor()

# Test diseases
TNF_DISEASES = {
    'MONDO:0015564': 'Castleman Disease (iMCD-TAFRO)',
    'MONDO:0008383': 'Rheumatoid Arthritis',
    'MONDO:0005011': 'Crohn Disease',
    'MONDO:0005083': 'Psoriasis'
}

NON_TNF_DISEASES = {
    'MONDO:0005148': 'Type 2 Diabetes',
    'MONDO:0005044': 'Hypertension',
    'MONDO:0011382': 'Alzheimer Disease'
}

# Train models once (like Phase 3)
print("\n" + "="*70)
print("TRAINING MODELS (ONCE)")
print("="*70)

print("\nBuilding baseline graph (3D)...")
baseline_data = predictor.build_graph_with_experimental_features(use_experimental=False)
baseline_model = GraphSAGEModel(input_dim=3, hidden_dim=64, output_dim=32)

print("Training baseline model...")
baseline_model, baseline_data = predictor.train_model(baseline_data, baseline_model, epochs=200)

print("\nBuilding enhanced graph (4D with TNF)...")
enhanced_data = predictor.build_graph_with_experimental_features(use_experimental=True)
enhanced_model = GraphSAGEModel(input_dim=4, hidden_dim=64, output_dim=32)

print("Training enhanced model...")
enhanced_model, enhanced_data = predictor.train_model(enhanced_data, enhanced_model, epochs=200)

# Test on TNF diseases
print("\n" + "="*70)
print("TESTING TNF-MEDIATED DISEASES (Should Improve)")
print("="*70)

tnf_results = []
for disease_id, disease_name in TNF_DISEASES.items():
    print(f"\n{disease_name} ({disease_id}):")
    
    # Get rankings
    baseline_ranking = predictor.evaluate_drug_ranking(baseline_model, baseline_data, disease_id)
    enhanced_ranking = predictor.evaluate_drug_ranking(enhanced_model, enhanced_data, disease_id)
    
    # Find adalimumab rank
    baseline_rank = None
    for rank, (drug, score) in enumerate(baseline_ranking.items(), 1):
        if drug == predictor.adalimumab_id:
            baseline_rank = rank
            break
    
    enhanced_rank = None
    for rank, (drug, score) in enumerate(enhanced_ranking.items(), 1):
        if drug == predictor.adalimumab_id:
            enhanced_rank = rank
            break
    
    if baseline_rank and enhanced_rank:
        improvement = baseline_rank - enhanced_rank
        tnf_results.append({
            'disease_id': disease_id,
            'disease_name': disease_name,
            'baseline_rank': baseline_rank,
            'enhanced_rank': enhanced_rank,
            'improvement': improvement
        })
        
        print(f"  Baseline: #{baseline_rank:,}")
        print(f"  Enhanced: #{enhanced_rank:,}")
        print(f"  Improvement: {improvement:+,} positions")
    else:
        print(f"  ⚠️  Adalimumab not found in rankings")

# Test on non-TNF diseases
print("\n" + "="*70)
print("TESTING NON-TNF DISEASES (Should NOT Improve Much)")
print("="*70)

non_tnf_results = []
for disease_id, disease_name in NON_TNF_DISEASES.items():
    print(f"\n{disease_name} ({disease_id}):")
    
    baseline_ranking = predictor.evaluate_drug_ranking(baseline_model, baseline_data, disease_id)
    enhanced_ranking = predictor.evaluate_drug_ranking(enhanced_model, enhanced_data, disease_id)
    
    baseline_rank = None
    for rank, (drug, score) in enumerate(baseline_ranking.items(), 1):
        if drug == predictor.adalimumab_id:
            baseline_rank = rank
            break
    
    enhanced_rank = None
    for rank, (drug, score) in enumerate(enhanced_ranking.items(), 1):
        if drug == predictor.adalimumab_id:
            enhanced_rank = rank
            break
    
    if baseline_rank and enhanced_rank:
        improvement = baseline_rank - enhanced_rank
        non_tnf_results.append({
            'disease_id': disease_id,
            'disease_name': disease_name,
            'baseline_rank': baseline_rank,
            'enhanced_rank': enhanced_rank,
            'improvement': improvement
        })
        
        print(f"  Baseline: #{baseline_rank:,}")
        print(f"  Enhanced: #{enhanced_rank:,}")
        print(f"  Improvement: {improvement:+,} positions")
    else:
        print(f"  ⚠️  Adalimumab not found in rankings")

# Statistical comparison
print("\n" + "="*70)
print("STATISTICAL COMPARISON")
print("="*70)

tnf_improvements = [r['improvement'] for r in tnf_results]
non_tnf_improvements = [r['improvement'] for r in non_tnf_results]

tnf_mean = np.mean(tnf_improvements)
tnf_std = np.std(tnf_improvements, ddof=1) if len(tnf_improvements) > 1 else 0
non_tnf_mean = np.mean(non_tnf_improvements)
non_tnf_std = np.std(non_tnf_improvements, ddof=1) if len(non_tnf_improvements) > 1 else 0

print(f"\nTNF-Mediated Diseases:")
print(f"  Mean improvement: {tnf_mean:,.0f} ± {tnf_std:,.0f} positions")
print(f"  Improvements: {[f'{x:,}' for x in tnf_improvements]}")

print(f"\nNon-TNF Diseases:")
print(f"  Mean improvement: {non_tnf_mean:,.0f} ± {non_tnf_std:,.0f} positions")
print(f"  Improvements: {[f'{x:,}' for x in non_tnf_improvements]}")

# t-test if we have enough data
if len(tnf_improvements) > 1 and len(non_tnf_improvements) > 1:
    t_stat, p_value = stats.ttest_ind(tnf_improvements, non_tnf_improvements)
    
    print(f"\nIndependent t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    
    if p_value < 0.01:
        print(f"  ✅ HIGHLY SIGNIFICANT (p < 0.01)")
        print(f"  TNF diseases improve significantly more than non-TNF diseases")
    elif p_value < 0.05:
        print(f"  ✅ SIGNIFICANT (p < 0.05)")
        print(f"  TNF diseases improve more than non-TNF diseases")
    else:
        print(f"  ❌ NOT SIGNIFICANT (p >= 0.05)")
        print(f"  ⚠️  WARNING: No clear disease specificity detected")

# Interpretation
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if tnf_mean > non_tnf_mean * 5:  # TNF improves 5x more
    print("✅ STRONG DISEASE SPECIFICITY:")
    print(f"   TNF diseases improve {tnf_mean/non_tnf_mean:.1f}x more than non-TNF diseases")
    print("   This supports the biological mechanism of TNF feature")
elif tnf_mean > non_tnf_mean * 2:
    print("⚠️  MODERATE DISEASE SPECIFICITY:")
    print(f"   TNF diseases improve {tnf_mean/non_tnf_mean:.1f}x more than non-TNF diseases")
else:
    print("❌ WEAK OR NO DISEASE SPECIFICITY:")
    print("   TNF diseases don't improve much more than non-TNF diseases")
    print("   This suggests the feature may not be disease-specific")

# Save results
results_dir = Path('results/phase_4_2_specificity')
results_dir.mkdir(parents=True, exist_ok=True)

results = {
    'tnf_mediated_diseases': tnf_results,
    'non_tnf_diseases': non_tnf_results,
    'summary': {
        'tnf_mean_improvement': float(tnf_mean),
        'tnf_std': float(tnf_std),
        'non_tnf_mean_improvement': float(non_tnf_mean),
        'non_tnf_std': float(non_tnf_std),
        't_statistic': float(t_stat) if len(tnf_improvements) > 1 and len(non_tnf_improvements) > 1 else None,
        'p_value': float(p_value) if len(tnf_improvements) > 1 and len(non_tnf_improvements) > 1 else None,
        'fold_difference': float(tnf_mean / non_tnf_mean) if non_tnf_mean != 0 else None
    }
}

with open(results_dir / 'disease_specificity_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("✅ PHASE 4.2 COMPLETE")
print(f"{'='*70}")
print(f"\nResults saved to: {results_dir}")

PYEOF
