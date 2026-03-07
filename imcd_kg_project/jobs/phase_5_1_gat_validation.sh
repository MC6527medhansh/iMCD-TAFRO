#!/bin/bash
#SBATCH --job-name=phase_5_1_gat
#SBATCH --account=st-singha53-1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/phase_5_1_gat_%j.txt

# phase_5_1_gat_validation.sh
# ----------------------------
# Purpose: End-to-end validation of GATConv + edge weights architecture.
#
# Three experiments in one job:
#   Exp A: GATConv + uniform weights (new baseline — replaces GraphSAGE baseline)
#   Exp B: GATConv + random weights on Castleman->protein edges
#           Validation: if adalimumab rank differs from Exp A, GATConv reads edge_attr
#   Exp C: GATConv + TNF weight=30.7, all others=1.0
#           Validation: adalimumab should rank higher for TNF diseases than non-TNF
#
# Key questions answered:
#   1. Do edge weights affect drug rankings at all? (A vs B)
#   2. Is the effect disease-specific? (C, TNF vs non-TNF diseases)
#   3. Is the approach stable across seeds? (3 seeds per experiment)
#
# Results saved to: results/phase_5_1/
#   - exp_a_uniform.json
#   - exp_b_random.json
#   - exp_c_tnf.json
#   - summary.json

module load gcc/9.4.0 miniconda3/4.9.2
source activate imcd_kg

cd /scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project

echo "========================================================================"
echo "PHASE 5.1: GATConv + Edge Weights Architecture Validation"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "========================================================================"

python - << 'PYEOF'
import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
import json
from pathlib import Path

from enhanced_kgnn.gat_predictor import GATModel, GATPredictor

print("=" * 70)
print("PHASE 5.1: GATConv + Edge Weights Architecture Validation")
print("=" * 70)
print()
print("Architecture rationale:")
print("  - GraphSAGE node features FAILED: non-TNF diseases improved MORE")
print("    than TNF diseases (fold diff=0.88, not disease-specific)")
print("  - GATConv edge weights: weight on TNF->Castleman edge pulls")
print("    Castleman embedding toward TNF -> toward adalimumab")
print("  - Disease-specific by construction: other diseases keep weight=1.0")
print()

# -----------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------

SEEDS = [42, 123, 303]
EPOCHS = 200

TNF_DISEASES = {
    'MONDO:0015564': 'Castleman (iMCD-TAFRO)',
    'MONDO:0008383': 'Rheumatoid Arthritis',
    'MONDO:0005011': 'Crohn Disease',
    'MONDO:0005083': 'Psoriasis',
}

NON_TNF_DISEASES = {
    'MONDO:0005148': 'Type 2 Diabetes',
    'MONDO:0005044': 'Hypertension',
    'MONDO:0011382': 'Alzheimer Disease',
}

results_dir = Path('results/phase_5_1')
results_dir.mkdir(parents=True, exist_ok=True)

predictor = GATPredictor()

# Load graph once to get Castleman's protein neighbors
# (neighbors do not change between experiments)
print("Loading graph to identify Castleman protein neighbors...")
data_tmp = predictor.build_graph()
castleman_neighbors = predictor.get_disease_protein_neighbors(predictor.castleman_id)
print(f"Castleman protein neighbors found: {len(castleman_neighbors)}")

tnf_is_neighbor = predictor.tnf_id in castleman_neighbors
print(f"TNF is direct neighbor of Castleman: {tnf_is_neighbor}")
if not tnf_is_neighbor:
    print("WARNING: TNF not a direct neighbor. Edge weight cannot be set on "
          "TNF->Castleman edge. This would invalidate the experiment.")
    print("Aborting.")
    sys.exit(1)

print()


def run_experiment(experiment_name, disease_edge_weights, seeds, epochs):
    """
    Run GATConv training and evaluation for one experiment across multiple seeds.

    Returns:
        dict with per-seed and aggregated results for TNF and non-TNF diseases
    """
    print(f"{'=' * 70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'=' * 70}")

    if disease_edge_weights is None:
        print("Edge weights: UNIFORM (all 1.0) — baseline")
    else:
        tnf_w = disease_edge_weights.get(predictor.tnf_id, "N/A")
        vals = list(disease_edge_weights.values())
        print(f"Edge weights on Castleman neighbors: {len(disease_edge_weights)} proteins")
        print(f"  TNF weight: {tnf_w}")
        print(f"  Range: [{min(vals):.3f}, {max(vals):.3f}]")

    print(f"Seeds: {seeds}, Epochs: {epochs}")
    print()

    seed_results = []

    for seed in seeds:
        print(f"  --- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Build graph with this experiment's edge weights
        data = predictor.build_graph(disease_edge_weights=disease_edge_weights)

        # Train GATConv (3D input features, hidden=64, output=32, 4 heads)
        model = GATModel(input_dim=3, hidden_dim=64, output_dim=32,
                         heads=4, edge_dim=1, dropout=0.1)
        model = predictor.train_model(data, model, epochs=epochs, lr=0.01)

        # Evaluate adalimumab rank for all diseases
        tnf_results = []
        for disease_id, disease_name in TNF_DISEASES.items():
            rank, score = predictor.get_adalimumab_rank(model, data, disease_id)
            tnf_results.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'rank': rank,
                'score': float(score) if score is not None else None,
            })
            status = f"#{rank:,}" if rank else "NOT FOUND"
            print(f"    [TNF]     {disease_name:30} adalimumab rank: {status}")

        non_tnf_results = []
        for disease_id, disease_name in NON_TNF_DISEASES.items():
            rank, score = predictor.get_adalimumab_rank(model, data, disease_id)
            non_tnf_results.append({
                'disease_id': disease_id,
                'disease_name': disease_name,
                'rank': rank,
                'score': float(score) if score is not None else None,
            })
            status = f"#{rank:,}" if rank else "NOT FOUND"
            print(f"    [non-TNF] {disease_name:30} adalimumab rank: {status}")

        seed_results.append({
            'seed': seed,
            'tnf': tnf_results,
            'non_tnf': non_tnf_results,
        })
        print()

    # Aggregate: mean rank per disease across seeds
    def aggregate_ranks(results_list, disease_key):
        """Compute mean/std of adalimumab rank across seeds for each disease."""
        by_disease = {}
        for seed_res in results_list:
            for d in seed_res[disease_key]:
                did = d['disease_id']
                if did not in by_disease:
                    by_disease[did] = {
                        'name': d['disease_name'],
                        'ranks': [],
                        'scores': [],
                    }
                if d['rank'] is not None:
                    by_disease[did]['ranks'].append(d['rank'])
                if d['score'] is not None:
                    by_disease[did]['scores'].append(d['score'])
        agg = {}
        for did, vals in by_disease.items():
            ranks = vals['ranks']
            agg[did] = {
                'name': vals['name'],
                'mean_rank': float(np.mean(ranks)) if ranks else None,
                'std_rank': float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
                'ranks': ranks,
            }
        return agg

    tnf_agg = aggregate_ranks(seed_results, 'tnf')
    non_tnf_agg = aggregate_ranks(seed_results, 'non_tnf')

    print(f"  SUMMARY — {experiment_name}")
    print(f"  {'Disease':<35} {'Mean Rank':>12} {'Std':>8}")
    print(f"  {'-------':<35} {'---------':>12} {'---':>8}")
    for did, v in tnf_agg.items():
        mr = v['mean_rank']
        s = v['std_rank']
        tag = "[TNF]"
        print(f"  {tag} {v['name']:<29} {mr:>12,.0f} {s:>8,.0f}")
    for did, v in non_tnf_agg.items():
        mr = v['mean_rank']
        s = v['std_rank']
        tag = "[non-TNF]"
        print(f"  {tag} {v['name']:<25} {mr:>12,.0f} {s:>8,.0f}")

    castleman_rank = tnf_agg.get('MONDO:0015564', {}).get('mean_rank')
    print(f"\n  Adalimumab mean rank for Castleman: "
          f"{castleman_rank:,.0f}" if castleman_rank else "  Castleman rank: N/A")
    print()

    return {
        'experiment': experiment_name,
        'seeds': seeds,
        'epochs': epochs,
        'seed_results': seed_results,
        'tnf_aggregate': tnf_agg,
        'non_tnf_aggregate': non_tnf_agg,
    }


# -----------------------------------------------------------------------
# Experiment A: Uniform weights (GATConv baseline)
# -----------------------------------------------------------------------

exp_a = run_experiment(
    experiment_name="A: GATConv + Uniform Weights (Baseline)",
    disease_edge_weights=None,
    seeds=SEEDS,
    epochs=EPOCHS,
)
with open(results_dir / 'exp_a_uniform.json', 'w') as f:
    json.dump(exp_a, f, indent=2)
print("Saved: results/phase_5_1/exp_a_uniform.json")


# -----------------------------------------------------------------------
# Experiment B: Random weights on Castleman->protein edges
# Architecture sanity check: if rankings shift vs Exp A, edge_attr works
# -----------------------------------------------------------------------

random_weights = predictor.make_random_weights(
    castleman_neighbors,
    seed=42,
    weight_min=0.5,
    weight_max=5.0,
)
exp_b = run_experiment(
    experiment_name="B: GATConv + Random Edge Weights (Architecture Validation)",
    disease_edge_weights=random_weights,
    seeds=SEEDS,
    epochs=EPOCHS,
)
with open(results_dir / 'exp_b_random.json', 'w') as f:
    json.dump(exp_b, f, indent=2)
print("Saved: results/phase_5_1/exp_b_random.json")


# -----------------------------------------------------------------------
# Experiment C: TNF fold-change weight (real biological signal)
# Key test: adalimumab should rank higher for TNF diseases than non-TNF
# -----------------------------------------------------------------------

tnf_weights = predictor.make_tnf_weights(
    castleman_neighbors,
    tnf_weight=predictor.tnf_fold_change,  # 30.7x
    other_weight=1.0,
)
exp_c = run_experiment(
    experiment_name="C: GATConv + TNF Fold-Change Weight (Real Signal)",
    disease_edge_weights=tnf_weights,
    seeds=SEEDS,
    epochs=EPOCHS,
)
with open(results_dir / 'exp_c_tnf.json', 'w') as f:
    json.dump(exp_c, f, indent=2)
print("Saved: results/phase_5_1/exp_c_tnf.json")


# -----------------------------------------------------------------------
# Compare experiments and answer the key questions
# -----------------------------------------------------------------------

print()
print("=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

def castleman_mean_rank(exp):
    return exp['tnf_aggregate']['MONDO:0015564']['mean_rank']

rank_a = castleman_mean_rank(exp_a)
rank_b = castleman_mean_rank(exp_b)
rank_c = castleman_mean_rank(exp_c)

print(f"\nAdalimumab rank for Castleman disease:")
print(f"  Exp A (uniform weights):     #{rank_a:>10,.0f}")
print(f"  Exp B (random weights):      #{rank_b:>10,.0f}")
print(f"  Exp C (TNF fold-change):     #{rank_c:>10,.0f}")

# Question 1: Do edge weights affect rankings?
rank_diff_ab = abs(rank_a - rank_b) if rank_a and rank_b else 0
if rank_diff_ab > 1000:
    q1 = "YES — GATConv responds to edge_attr"
else:
    q1 = "WEAK — edge weights have minimal effect (investigate)"
print(f"\nQ1 (Do edge weights affect rankings?): {q1}")
print(f"     Rank change A->B: {rank_a - rank_b:+,.0f}")

# Question 2: Is Exp C disease-specific?
tnf_ranks_c = [
    v['mean_rank']
    for v in exp_c['tnf_aggregate'].values()
    if v['mean_rank'] is not None
]
non_tnf_ranks_c = [
    v['mean_rank']
    for v in exp_c['non_tnf_aggregate'].values()
    if v['mean_rank'] is not None
]

if tnf_ranks_c and non_tnf_ranks_c:
    mean_tnf = np.mean(tnf_ranks_c)
    mean_non_tnf = np.mean(non_tnf_ranks_c)
    # Lower rank = better (rank 1 is best)
    # We want adalimumab ranked higher (lower number) for TNF diseases
    if mean_tnf < mean_non_tnf:
        q2 = "YES — adalimumab ranks higher for TNF diseases than non-TNF"
        specificity = "DISEASE-SPECIFIC"
    else:
        q2 = "NO — adalimumab ranks similarly or worse for TNF diseases"
        specificity = "NOT DISEASE-SPECIFIC"
    print(f"\nQ2 (Disease specificity in Exp C?): {q2}")
    print(f"     TNF diseases mean rank:     #{mean_tnf:>10,.0f}")
    print(f"     Non-TNF diseases mean rank: #{mean_non_tnf:>10,.0f}")
    print(f"     Verdict: {specificity}")
else:
    q2 = "Could not evaluate (missing ranks)"
    specificity = "UNKNOWN"
    mean_tnf = None
    mean_non_tnf = None

# Save summary
summary = {
    'castleman_ranks': {
        'exp_a_uniform': rank_a,
        'exp_b_random': rank_b,
        'exp_c_tnf': rank_c,
    },
    'q1_edge_weights_effective': q1,
    'q1_rank_change_a_to_b': float(rank_a - rank_b) if rank_a and rank_b else None,
    'q2_disease_specific': q2,
    'q2_specificity_verdict': specificity,
    'q2_tnf_mean_rank': float(mean_tnf) if mean_tnf else None,
    'q2_non_tnf_mean_rank': float(mean_non_tnf) if mean_non_tnf else None,
    'castleman_protein_neighbors': len(castleman_neighbors),
    'tnf_is_direct_neighbor': tnf_is_neighbor,
    'tnf_fold_change_used': float(predictor.tnf_fold_change),
    'seeds': SEEDS,
    'epochs': EPOCHS,
}

with open(results_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print()
print("=" * 70)
print("PHASE 5.1 COMPLETE")
print("=" * 70)
print()
print(f"Results saved to: results/phase_5_1/")
print(f"  exp_a_uniform.json  — GATConv baseline (uniform weights)")
print(f"  exp_b_random.json   — random weights (architecture validation)")
print(f"  exp_c_tnf.json      — TNF fold-change weights (real signal)")
print(f"  summary.json        — comparison + key questions answered")

PYEOF

echo "========================================================================"
echo "End: $(date)"
echo "========================================================================"
