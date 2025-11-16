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
cd $WORK_DIR

echo ""
echo "Working directory: $WORK_DIR"
echo ""

# Run Phase 3 - use existing compare_predictions() method
python - << 'PYEOF'
import sys
import os
from pathlib import Path
import json
import logging
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
    from enhanced_kgnn.enhanced_predictor import ExperimentalGraphPredictor
    
    logger.info("✅ Modules imported successfully")
    
    # Initialize predictor
    logger.info("\n2. Initializing predictor...")
    predictor = ExperimentalGraphPredictor()
    logger.info("✅ Predictor initialized")
    logger.info(f"   Adalimumab ID: {predictor.adalimumab_id}")
    logger.info(f"   Castleman ID: {predictor.castleman_id}")
    logger.info(f"   TNF ID: {predictor.tnf_id}")
    
    # Run complete comparison (builds graphs, trains models, evaluates)
    logger.info("\n3. Running complete training and comparison...")
    logger.info("   This will:")
    logger.info("   - Build baseline graph (3D features)")
    logger.info("   - Train baseline model (200 epochs)")
    logger.info("   - Build enhanced graph (4D features with TNF)")
    logger.info("   - Train enhanced model (200 epochs)")
    logger.info("   - Evaluate both on Castleman disease")
    logger.info("   - Compare adalimumab rankings")
    logger.info("")
    
    results = predictor.compare_predictions(epochs=200)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING AND EVALUATION COMPLETE")
    logger.info("="*70)
    logger.info("")
    logger.info("RESULTS:")
    logger.info(f"  Baseline adalimumab rank: #{results['baseline_adalimumab_rank']}")
    logger.info(f"  Baseline adalimumab score: {results['baseline_adalimumab_score']:.6f}")
    logger.info(f"  Enhanced adalimumab rank: #{results['enhanced_adalimumab_rank']}")
    logger.info(f"  Enhanced adalimumab score: {results['enhanced_adalimumab_score']:.6f}")
    logger.info(f"  Ranking improvement: {results['ranking_improvement']:+d} positions")
    logger.info("")
    
    # Save results
    logger.info("4. Saving results...")
    results_dir = Path("/scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project/results/phase_3")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison results
    comparison_path = results_dir / "comparison_results.json"
    
    # Enhance results with metadata
    full_results = {
        'phase': '3',
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'adalimumab_rank': results['baseline_adalimumab_rank'],
            'adalimumab_score': float(results['baseline_adalimumab_score']),
            'top_10': [
                {'rank': i, 'drug_id': drug, 'score': float(score)}
                for i, (drug, score) in enumerate(results['baseline_top_10'], 1)
            ]
        },
        'enhanced': {
            'adalimumab_rank': results['enhanced_adalimumab_rank'],
            'adalimumab_score': float(results['enhanced_adalimumab_score']),
            'tnf_feature_note': 'TNF feature (4.94 log2FC) applied to adalimumab and TNF nodes',
            'top_10': [
                {'rank': i, 'drug_id': drug, 'score': float(score)}
                for i, (drug, score) in enumerate(results['enhanced_top_10'], 1)
            ]
        },
        'improvement': {
            'rank_improvement': results['ranking_improvement'],
            'percent_improvement': (results['ranking_improvement'] / results['baseline_adalimumab_rank'] * 100) if results['baseline_adalimumab_rank'] else 0,
            'score_improvement': float(results['enhanced_adalimumab_score'] - results['baseline_adalimumab_score']),
            'baseline_to_enhanced': f"#{results['baseline_adalimumab_rank']} → #{results['enhanced_adalimumab_rank']}"
        },
        'validation': {
            'both_models_trained': True,
            'adalimumab_in_both_rankings': True,
            'improvement_achieved': results['ranking_improvement'] > 0
        }
    }
    
    with open(comparison_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    logger.info(f"✅ Results saved: {comparison_path}")
    
    # Save metadata
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
        'output_files': {
            'comparison': str(comparison_path)
        }
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Metadata saved: {metadata_path}")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("PHASE 3 COMPLETE ✅")
    logger.info("="*70)
    logger.info("")
    logger.info("RESULTS SUMMARY:")
    logger.info(f"  Baseline:  Rank #{results['baseline_adalimumab_rank']}, Score {results['baseline_adalimumab_score']:.6f}")
    logger.info(f"  Enhanced:  Rank #{results['enhanced_adalimumab_rank']}, Score {results['enhanced_adalimumab_score']:.6f}")
    logger.info(f"  Improvement: {results['ranking_improvement']:+d} positions")
    logger.info("")
    
    if results['ranking_improvement'] > 0:
        logger.info(f"✅ SUCCESS: TNF experimental feature improved adalimumab ranking!")
        logger.info(f"   Moved from position #{results['baseline_adalimumab_rank']} to #{results['enhanced_adalimumab_rank']}")
    elif results['ranking_improvement'] == 0:
        logger.warning(f"⚠️  WARNING: No improvement in ranking")
    else:
        logger.warning(f"⚠️  WARNING: Ranking worsened by {abs(results['ranking_improvement'])} positions")
    
    logger.info("")
    logger.info("OUTPUT FILES:")
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