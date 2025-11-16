#!/bin/bash
#SBATCH --job-name=phase_3
#SBATCH --account=st-singha53-1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/phase_3_%j.txt

echo "========================================================================"
echo "Phase 3: Model Training and Comparison"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================================================"

# Load environment
module load gcc/9.4.0 miniconda3/4.9.2
source activate imcd_kg

# Set paths
WORK_DIR="/scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project"
DATA_ROOT="/arc/project/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project/data"

cd $WORK_DIR
export KG_ROOT="${DATA_ROOT}/kgml_data/bkg_rtxkg2c_v2.7.3"

echo ""
echo "Working directory: $WORK_DIR"
echo "Data root: $DATA_ROOT"
echo ""

# Run Phase 3
python - << 'PYEOF'
import sys
import os
from pathlib import Path
import json
import logging
import torch
import numpy as np
from datetime import datetime

# Setup paths
project_root = Path("/scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project")
sys.path.insert(0, str(project_root / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("="*70)
logger.info("PHASE 3: MODEL TRAINING AND COMPARISON")
logger.info("="*70)

try:
    # Import modules
    logger.info("\n1. Importing modules...")
    from enhanced_kgnn.enhanced_predictor import ExperimentalGraphPredictor, GraphSAGEModel
    from config import (
        PROCESSED_GRAPH_PATH, 
        ADALIMUMAB_ID, 
        CASTLEMAN_ID, 
        TNF_ID,
        DATA_DIR
    )
    
    logger.info("✅ Modules imported successfully")
    
    # Validate inputs exist
    logger.info("\n2. Validating input files...")
    
    baseline_graph_path = DATA_DIR / "processed" / "baseline_graph.pt"
    enhanced_graph_path = DATA_DIR / "processed" / "enhanced_graph.pt"
    
    if not baseline_graph_path.exists():
        raise FileNotFoundError(f"Baseline graph not found: {baseline_graph_path}")
    if not enhanced_graph_path.exists():
        raise FileNotFoundError(f"Enhanced graph not found: {enhanced_graph_path}")
    
    baseline_size_mb = baseline_graph_path.stat().st_size / 1e6
    enhanced_size_mb = enhanced_graph_path.stat().st_size / 1e6
    
    logger.info(f"✅ Baseline graph found: {baseline_graph_path}")
    logger.info(f"   Size: {baseline_size_mb:.2f} MB")
    logger.info(f"✅ Enhanced graph found: {enhanced_graph_path}")
    logger.info(f"   Size: {enhanced_size_mb:.2f} MB")
    
    # Initialize predictor
    logger.info("\n3. Initializing predictor...")
    predictor = ExperimentalGraphPredictor()
    logger.info("✅ Predictor initialized")
    logger.info(f"   Adalimumab ID: {predictor.adalimumab_id}")
    logger.info(f"   Castleman ID: {predictor.castleman_id}")
    logger.info(f"   TNF ID: {predictor.tnf_id}")
    
    # Load graphs
    logger.info("\n4. Loading graphs...")
    logger.info("   Loading baseline graph...")
    baseline_data = torch.load(baseline_graph_path)
    logger.info(f"   ✅ Baseline: {baseline_data.num_nodes:,} nodes, {baseline_data.edge_index.shape[1]:,} edges, {baseline_data.x.shape[1]}D features")
    
    logger.info("   Loading enhanced graph...")
    enhanced_data = torch.load(enhanced_graph_path)
    logger.info(f"   ✅ Enhanced: {enhanced_data.num_nodes:,} nodes, {enhanced_data.edge_index.shape[1]:,} edges, {enhanced_data.x.shape[1]}D features")
    
    # Validate graphs loaded correctly
    assert baseline_data.x.shape[1] == 3, f"Baseline should have 3D features, got {baseline_data.x.shape[1]}D"
    assert enhanced_data.x.shape[1] == 4, f"Enhanced should have 4D features, got {enhanced_data.x.shape[1]}D"
    assert baseline_data.num_nodes == enhanced_data.num_nodes, "Node count mismatch between graphs"
    logger.info("   ✅ Graph validation passed")
    
    # Set random seeds for reproducibility
    logger.info("\n5. Setting random seeds for reproducibility...")
    torch.manual_seed(42)
    np.random.seed(42)
    logger.info("   ✅ Seeds set: 42")
    
    # ========================================================================
    # TRAIN BASELINE MODEL
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("6. TRAINING BASELINE MODEL (3D features)")
    logger.info("="*70)
    
    baseline_model = GraphSAGEModel(input_dim=3, hidden_dim=64, output_dim=32, dropout=0.1)
    num_params_baseline = sum(p.numel() for p in baseline_model.parameters())
    logger.info(f"   Model created: {num_params_baseline:,} parameters")
    logger.info(f"   Architecture: 3D → 64D → 32D")
    logger.info(f"   Dropout: 0.1")
    logger.info("")
    
    logger.info("   Starting training (200 epochs)...")
    baseline_model, baseline_data_trained = predictor.train_model(
        baseline_data, 
        baseline_model, 
        epochs=200,
        lr=0.01
    )
    logger.info("   ✅ Baseline model training complete")
    
    # Validate baseline model
    logger.info("\n   Validating baseline model...")
    baseline_model.eval()
    with torch.no_grad():
        z_baseline = baseline_model(baseline_data_trained.x, baseline_data_trained.edge_index)
        assert not torch.isnan(z_baseline).any(), "NaN in baseline embeddings"
        assert not torch.isinf(z_baseline).any(), "Inf in baseline embeddings"
    logger.info("   ✅ Baseline model validation passed")
    
    # ========================================================================
    # TRAIN ENHANCED MODEL
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("7. TRAINING ENHANCED MODEL (4D features with TNF)")
    logger.info("="*70)
    
    # Reset seeds for fair comparison
    torch.manual_seed(42)
    np.random.seed(42)
    
    enhanced_model = GraphSAGEModel(input_dim=4, hidden_dim=64, output_dim=32, dropout=0.1)
    num_params_enhanced = sum(p.numel() for p in enhanced_model.parameters())
    logger.info(f"   Model created: {num_params_enhanced:,} parameters")
    logger.info(f"   Architecture: 4D → 64D → 32D")
    logger.info(f"   Dropout: 0.1")
    
    # Verify TNF feature is present
    tnf_idx = predictor.entity_to_idx[TNF_ID]
    tnf_feature = enhanced_data.x[tnf_idx, 3].item()
    logger.info(f"   TNF feature verified: {tnf_feature:.4f} at node {tnf_idx}")
    logger.info("")
    
    logger.info("   Starting training (200 epochs)...")
    enhanced_model, enhanced_data_trained = predictor.train_model(
        enhanced_data,
        enhanced_model,
        epochs=200,
        lr=0.01
    )
    logger.info("   ✅ Enhanced model training complete")
    
    # Validate enhanced model
    logger.info("\n   Validating enhanced model...")
    enhanced_model.eval()
    with torch.no_grad():
        z_enhanced = enhanced_model(enhanced_data_trained.x, enhanced_data_trained.edge_index)
        assert not torch.isnan(z_enhanced).any(), "NaN in enhanced embeddings"
        assert not torch.isinf(z_enhanced).any(), "Inf in enhanced embeddings"
    logger.info("   ✅ Enhanced model validation passed")
    
    # ========================================================================
    # EVALUATE RANKINGS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("8. EVALUATING DRUG RANKINGS FOR CASTLEMAN DISEASE")
    logger.info("="*70)
    
    logger.info("\n   Evaluating baseline model...")
    baseline_ranking = predictor.evaluate_drug_ranking(
        baseline_model, 
        baseline_data_trained, 
        predictor.castleman_id
    )
    logger.info(f"   ✅ Ranked {len(baseline_ranking):,} drugs")
    
    logger.info("\n   Evaluating enhanced model...")
    enhanced_ranking = predictor.evaluate_drug_ranking(
        enhanced_model,
        enhanced_data_trained,
        predictor.castleman_id
    )
    logger.info(f"   ✅ Ranked {len(enhanced_ranking):,} drugs")
    
    # Validate both rankings have adalimumab
    assert predictor.adalimumab_id in baseline_ranking, "Adalimumab not in baseline ranking"
    assert predictor.adalimumab_id in enhanced_ranking, "Adalimumab not in enhanced ranking"
    logger.info("   ✅ Adalimumab present in both rankings")
    
    # ========================================================================
    # FIND ADALIMUMAB RANKINGS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("9. ANALYZING ADALIMUMAB RANKINGS")
    logger.info("="*70)
    
    # Find baseline rank
    baseline_adalimumab_rank = None
    baseline_adalimumab_score = 0.0
    for rank, (drug, score) in enumerate(baseline_ranking.items(), 1):
        if drug == predictor.adalimumab_id:
            baseline_adalimumab_rank = rank
            baseline_adalimumab_score = score
            break
    
    # Find enhanced rank
    enhanced_adalimumab_rank = None
    enhanced_adalimumab_score = 0.0
    for rank, (drug, score) in enumerate(enhanced_ranking.items(), 1):
        if drug == predictor.adalimumab_id:
            enhanced_adalimumab_rank = rank
            enhanced_adalimumab_score = score
            break
    
    # Calculate improvement
    if baseline_adalimumab_rank and enhanced_adalimumab_rank:
        improvement = baseline_adalimumab_rank - enhanced_adalimumab_rank
        percent_improvement = (improvement / baseline_adalimumab_rank) * 100
    else:
        raise ValueError("Could not find adalimumab in rankings")
    
    logger.info("")
    logger.info("   BASELINE MODEL:")
    logger.info(f"     Adalimumab rank: #{baseline_adalimumab_rank}")
    logger.info(f"     Adalimumab score: {baseline_adalimumab_score:.6f}")
    logger.info("")
    logger.info("   ENHANCED MODEL:")
    logger.info(f"     Adalimumab rank: #{enhanced_adalimumab_rank}")
    logger.info(f"     Adalimumab score: {enhanced_adalimumab_score:.6f}")
    logger.info("")
    logger.info("   IMPROVEMENT:")
    logger.info(f"     Position change: {improvement:+d} positions")
    logger.info(f"     Percent improvement: {percent_improvement:+.1f}%")
    logger.info(f"     Score increase: {enhanced_adalimumab_score - baseline_adalimumab_score:+.6f}")
    
    # Extract top 10 from each ranking
    baseline_top_10 = list(baseline_ranking.items())[:10]
    enhanced_top_10 = list(enhanced_ranking.items())[:10]
    
    logger.info("\n   BASELINE TOP 10:")
    for i, (drug, score) in enumerate(baseline_top_10, 1):
        marker = " ⭐" if drug == predictor.adalimumab_id else ""
        logger.info(f"     #{i}: {drug[:40]}... Score: {score:.6f}{marker}")
    
    logger.info("\n   ENHANCED TOP 10:")
    for i, (drug, score) in enumerate(enhanced_top_10, 1):
        marker = " ⭐" if drug == predictor.adalimumab_id else ""
        logger.info(f"     #{i}: {drug[:40]}... Score: {score:.6f}{marker}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("10. SAVING RESULTS")
    logger.info("="*70)
    
    results_dir = Path("/scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project/results/phase_3")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    logger.info("\n   Saving models...")
    baseline_model_path = results_dir / "baseline_model.pt"
    enhanced_model_path = results_dir / "enhanced_model.pt"
    
    torch.save(baseline_model.state_dict(), baseline_model_path)
    logger.info(f"   ✅ Baseline model: {baseline_model_path}")
    logger.info(f"      Size: {baseline_model_path.stat().st_size / 1e6:.2f} MB")
    
    torch.save(enhanced_model.state_dict(), enhanced_model_path)
    logger.info(f"   ✅ Enhanced model: {enhanced_model_path}")
    logger.info(f"      Size: {enhanced_model_path.stat().st_size / 1e6:.2f} MB")
    
    # Save rankings
    logger.info("\n   Saving rankings...")
    
    baseline_ranking_path = results_dir / "baseline_castleman_ranking.json"
    with open(baseline_ranking_path, 'w') as f:
        # Convert to serializable format
        ranking_dict = {
            'disease_id': predictor.castleman_id,
            'model': 'baseline',
            'total_drugs': len(baseline_ranking),
            'adalimumab_rank': baseline_adalimumab_rank,
            'adalimumab_score': baseline_adalimumab_score,
            'top_100': [
                {'rank': i, 'drug_id': drug, 'score': float(score)}
                for i, (drug, score) in enumerate(list(baseline_ranking.items())[:100], 1)
            ]
        }
        json.dump(ranking_dict, f, indent=2)
    logger.info(f"   ✅ Baseline ranking: {baseline_ranking_path}")
    
    enhanced_ranking_path = results_dir / "enhanced_castleman_ranking.json"
    with open(enhanced_ranking_path, 'w') as f:
        ranking_dict = {
            'disease_id': predictor.castleman_id,
            'model': 'enhanced',
            'total_drugs': len(enhanced_ranking),
            'adalimumab_rank': enhanced_adalimumab_rank,
            'adalimumab_score': enhanced_adalimumab_score,
            'top_100': [
                {'rank': i, 'drug_id': drug, 'score': float(score)}
                for i, (drug, score) in enumerate(list(enhanced_ranking.items())[:100], 1)
            ]
        }
        json.dump(ranking_dict, f, indent=2)
    logger.info(f"   ✅ Enhanced ranking: {enhanced_ranking_path}")
    
    # Save comprehensive comparison
    logger.info("\n   Saving comparison results...")
    comparison_path = results_dir / "comparison_results.json"
    
    comparison_results = {
        'phase': '3',
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'model_path': str(baseline_model_path),
            'adalimumab_rank': baseline_adalimumab_rank,
            'adalimumab_score': float(baseline_adalimumab_score),
            'total_drugs_ranked': len(baseline_ranking),
            'top_10': [
                {'rank': i, 'drug_id': drug, 'score': float(score)}
                for i, (drug, score) in enumerate(baseline_top_10, 1)
            ]
        },
        'enhanced': {
            'model_path': str(enhanced_model_path),
            'adalimumab_rank': enhanced_adalimumab_rank,
            'adalimumab_score': float(enhanced_adalimumab_score),
            'total_drugs_ranked': len(enhanced_ranking),
            'tnf_feature_value': float(tnf_feature),
            'top_10': [
                {'rank': i, 'drug_id': drug, 'score': float(score)}
                for i, (drug, score) in enumerate(enhanced_top_10, 1)
            ]
        },
        'improvement': {
            'rank_improvement': improvement,
            'percent_improvement': float(percent_improvement),
            'score_improvement': float(enhanced_adalimumab_score - baseline_adalimumab_score),
            'baseline_to_enhanced': f"#{baseline_adalimumab_rank} → #{enhanced_adalimumab_rank}"
        },
        'validation': {
            'both_models_trained': True,
            'adalimumab_in_both_rankings': True,
            'improvement_achieved': improvement > 0,
            'same_graph_structure': baseline_data.num_nodes == enhanced_data.num_nodes
        }
    }
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    logger.info(f"   ✅ Comparison results: {comparison_path}")
    
    # Save metadata
    logger.info("\n   Saving metadata...")
    metadata_path = results_dir / "metadata.json"
    metadata = {
        'phase': '3',
        'job_id': os.environ.get('SLURM_JOB_ID', 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'training_params': {
            'epochs': 200,
            'learning_rate': 0.01,
            'hidden_dim': 64,
            'output_dim': 32,
            'dropout': 0.1,
            'random_seed': 42
        },
        'input_files': {
            'baseline_graph': str(baseline_graph_path),
            'enhanced_graph': str(enhanced_graph_path),
            'baseline_graph_size_mb': float(baseline_size_mb),
            'enhanced_graph_size_mb': float(enhanced_size_mb)
        },
        'output_files': {
            'baseline_model': str(baseline_model_path),
            'enhanced_model': str(enhanced_model_path),
            'baseline_ranking': str(baseline_ranking_path),
            'enhanced_ranking': str(enhanced_ranking_path),
            'comparison': str(comparison_path)
        },
        'model_info': {
            'baseline_parameters': num_params_baseline,
            'enhanced_parameters': num_params_enhanced
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"   ✅ Metadata: {metadata_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 3 COMPLETE ✅")
    logger.info("="*70)
    logger.info("")
    logger.info("RESULTS SUMMARY:")
    logger.info(f"  Baseline:  Rank #{baseline_adalimumab_rank}, Score {baseline_adalimumab_score:.6f}")
    logger.info(f"  Enhanced:  Rank #{enhanced_adalimumab_rank}, Score {enhanced_adalimumab_score:.6f}")
    logger.info(f"  Improvement: {improvement:+d} positions ({percent_improvement:+.1f}%)")
    logger.info("")
    
    if improvement > 0:
        logger.info(f"✅ SUCCESS: TNF experimental feature improved adalimumab ranking!")
        logger.info(f"   Moved from position #{baseline_adalimumab_rank} to #{enhanced_adalimumab_rank}")
    elif improvement == 0:
        logger.warning(f"⚠️  WARNING: No improvement in ranking")
    else:
        logger.warning(f"⚠️  WARNING: Ranking worsened by {abs(improvement)} positions")
    
    logger.info("")
    logger.info("OUTPUT FILES:")
    logger.info(f"  Models:      {results_dir}/")
    logger.info(f"  Rankings:    {results_dir}/")
    logger.info(f"  Comparison:  {comparison_path}")
    logger.info(f"  Metadata:    {metadata_path}")
    logger.info("")
    logger.info("Ready for Phase 4: Validation & Testing")
    logger.info("="*70)
    
    # Exit successfully
    sys.exit(0)

except Exception as e:
    logger.error("="*70)
    logger.error("PHASE 3 FAILED ❌")
    logger.error("="*70)
    logger.error(f"Error: {e}")
    
    import traceback
    logger.error("\nFull traceback:")
    logger.error(traceback.format_exc())
    
    sys.exit(1)

PYEOF

echo ""
echo "========================================================================"
echo "Phase 3 Job Complete"
echo "End time: $(date)"
echo "========================================================================"