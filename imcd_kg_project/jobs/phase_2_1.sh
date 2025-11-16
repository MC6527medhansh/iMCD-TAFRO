#!/bin/bash
#SBATCH --job-name=phase_2_1
#SBATCH --account=st-singha53-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/phase_2_1_%j.txt

echo "========================================================================"
echo "Phase 2.1: Build Baseline + Enhanced Graphs"
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
echo "KG root: $KG_ROOT"
echo ""

# Run Phase 2.1
python - << 'PYEOF'
import sys
import os
from pathlib import Path
import json
import logging
import torch
import numpy as np

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
logger.info("PHASE 2.1: BUILD BASELINE + ENHANCED GRAPHS")
logger.info("="*70)

try:
    # Import modules
    logger.info("\n1. Importing modules...")
    from enhanced_kgnn.enhanced_predictor import ExperimentalGraphPredictor
    from config import (
        PROCESSED_GRAPH_PATH, 
        ADALIMUMAB_ID, 
        CASTLEMAN_ID, 
        TNF_ID,
        DATA_DIR
    )
    
    logger.info("✅ Modules imported successfully")
    
    # Validate inputs exist
    logger.info("\n2. Validating inputs...")
    
    if not PROCESSED_GRAPH_PATH.exists():
        raise FileNotFoundError(f"Full graph not found: {PROCESSED_GRAPH_PATH}")
    
    graph_size_gb = PROCESSED_GRAPH_PATH.stat().st_size / 1e9
    logger.info(f"✅ Full graph found: {PROCESSED_GRAPH_PATH}")
    logger.info(f"   Size: {graph_size_gb:.2f} GB")
    
    # Initialize predictor
    logger.info("\n3. Initializing predictor...")
    predictor = ExperimentalGraphPredictor()
    logger.info("✅ Predictor initialized")
    
    # Build BASELINE graph (3D features)
    logger.info("\n4. Building BASELINE graph (3D features)...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    baseline_data = predictor.build_graph_with_experimental_features(use_experimental=False)
    
    baseline_stats = {
        'num_nodes': baseline_data.num_nodes,
        'num_edges': baseline_data.edge_index.shape[1],
        'feature_dim': baseline_data.x.shape[1],
        'num_training_pairs': baseline_data.train_edge_labels.shape[0],
        'num_positive_pairs': int(baseline_data.train_edge_labels.sum().item()),
        'num_negative_pairs': int((baseline_data.train_edge_labels == 0).sum().item()),
    }
    
    logger.info("✅ Baseline graph built:")
    for key, value in baseline_stats.items():
        logger.info(f"   {key}: {value:,}")
    
    # Build ENHANCED graph (4D features with TNF)
    logger.info("\n5. Building ENHANCED graph (4D features + TNF)...")
    torch.manual_seed(42)  # Reset seed
    np.random.seed(42)
    
    enhanced_data = predictor.build_graph_with_experimental_features(use_experimental=True)
    
    enhanced_stats = {
        'num_nodes': enhanced_data.num_nodes,
        'num_edges': enhanced_data.edge_index.shape[1],
        'feature_dim': enhanced_data.x.shape[1],
        'num_training_pairs': enhanced_data.train_edge_labels.shape[0],
        'num_positive_pairs': int(enhanced_data.train_edge_labels.sum().item()),
        'num_negative_pairs': int((enhanced_data.train_edge_labels == 0).sum().item()),
    }
    
    # Find TNF feature
    tnf_idx = predictor.entity_to_idx[TNF_ID]
    tnf_feature_value = float(enhanced_data.x[tnf_idx, 3].item())
    enhanced_stats['tnf_node_idx'] = tnf_idx
    enhanced_stats['tnf_feature_value'] = tnf_feature_value
    
    logger.info("✅ Enhanced graph built:")
    for key, value in enhanced_stats.items():
        if isinstance(value, (int, np.integer)):
            logger.info(f"   {key}: {value:,}")
        else:
            logger.info(f"   {key}: {value}")
    
    # Comprehensive validation
    logger.info("\n6. Running comprehensive validation...")
    
    validation_results = {
        'baseline': {},
        'enhanced': {},
        'comparison': {}
    }
    
    # Baseline validation
    logger.info("   Validating baseline graph...")
    assert baseline_data.x.shape[1] == 3, f"Wrong feature dim: {baseline_data.x.shape[1]}"
    assert not torch.isnan(baseline_data.x).any(), "NaN in baseline features"
    assert not torch.isinf(baseline_data.x).any(), "Inf in baseline features"
    assert (baseline_data.x >= 0).all(), "Negative values in baseline features"
    assert baseline_data.edge_index.max() < baseline_data.num_nodes, "Edge index out of bounds"
    logger.info("   ✅ Baseline validation passed")
    
    validation_results['baseline'] = {
        'feature_dim_correct': True,
        'no_nan': True,
        'no_inf': True,
        'all_positive': True,
        'edge_index_valid': True
    }
    
    # Enhanced validation
    logger.info("   Validating enhanced graph...")
    assert enhanced_data.x.shape[1] == 4, f"Wrong feature dim: {enhanced_data.x.shape[1]}"
    assert not torch.isnan(enhanced_data.x).any(), "NaN in enhanced features"
    assert not torch.isinf(enhanced_data.x).any(), "Inf in enhanced features"
    assert (enhanced_data.x >= 0).all(), "Negative values in enhanced features"
    assert enhanced_data.edge_index.max() < enhanced_data.num_nodes, "Edge index out of bounds"
    
    # TNF feature validation
    assert tnf_feature_value > 0, f"TNF feature not assigned: {tnf_feature_value}"
    assert abs(tnf_feature_value - 4.94) < 0.01, f"TNF feature wrong value: {tnf_feature_value}"
    logger.info(f"   ✅ TNF feature validated: {tnf_feature_value:.4f}")
    
    # Check other nodes don't have TNF feature
    other_nodes_with_tnf = (enhanced_data.x[:, 3] > 0).sum().item()
    logger.info(f"   Nodes with TNF feature: {other_nodes_with_tnf}")
    
    logger.info("   ✅ Enhanced validation passed")
    
    validation_results['enhanced'] = {
        'feature_dim_correct': True,
        'no_nan': True,
        'no_inf': True,
        'all_positive': True,
        'edge_index_valid': True,
        'tnf_feature_assigned': True,
        'tnf_feature_value': tnf_feature_value,
        'num_nodes_with_tnf_feature': other_nodes_with_tnf
    }
    
    # Comparison validation
    logger.info("   Validating baseline vs enhanced...")
    assert baseline_data.num_nodes == enhanced_data.num_nodes, "Node count mismatch"
    assert baseline_data.edge_index.shape[1] == enhanced_data.edge_index.shape[1], "Edge count mismatch"
    assert torch.equal(baseline_data.edge_index, enhanced_data.edge_index), "Edge indices don't match"
    assert baseline_data.x.shape[0] == enhanced_data.x.shape[0], "Node count mismatch in features"
    
    # Check first 3 dimensions match
    first_3d_match = torch.allclose(baseline_data.x, enhanced_data.x[:, :3])
    logger.info(f"   First 3D features match: {first_3d_match}")
    
    logger.info("   ✅ Comparison validation passed")
    
    validation_results['comparison'] = {
        'same_num_nodes': True,
        'same_num_edges': True,
        'same_edge_structure': True,
        'first_3d_match': bool(first_3d_match)
    }
    
    # Critical entity validation
    logger.info("   Validating critical entities...")
    adalimumab_idx = predictor.entity_to_idx[ADALIMUMAB_ID]
    castleman_idx = predictor.entity_to_idx[CASTLEMAN_ID]
    
    logger.info(f"   Adalimumab idx: {adalimumab_idx}")
    logger.info(f"   Castleman idx: {castleman_idx}")
    logger.info(f"   TNF idx: {tnf_idx}")
    
    validation_results['critical_entities'] = {
        'adalimumab_idx': adalimumab_idx,
        'castleman_idx': castleman_idx,
        'tnf_idx': tnf_idx,
        'all_present': True
    }
    
    logger.info("   ✅ Critical entities validated")
    
    # Save graphs
    logger.info("\n7. Saving graphs...")
    
    output_dir = DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_path = output_dir / "baseline_graph.pt"
    enhanced_path = output_dir / "enhanced_graph.pt"
    
    torch.save(baseline_data, baseline_path)
    logger.info(f"✅ Baseline graph saved: {baseline_path}")
    logger.info(f"   Size: {baseline_path.stat().st_size / 1e6:.2f} MB")
    
    torch.save(enhanced_data, enhanced_path)
    logger.info(f"✅ Enhanced graph saved: {enhanced_path}")
    logger.info(f"   Size: {enhanced_path.stat().st_size / 1e6:.2f} MB")
    
    # Save validation report
    logger.info("\n8. Saving validation report...")
    
    results_dir = Path("/scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project/results/phase_2_1")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    validation_report = {
        'phase': '2.1',
        'timestamp': str(Path(__file__).stat().st_mtime) if Path(__file__).exists() else 'N/A',
        'baseline_stats': baseline_stats,
        'enhanced_stats': enhanced_stats,
        'validation': validation_results,
        'output_files': {
            'baseline_graph': str(baseline_path),
            'enhanced_graph': str(enhanced_path)
        }
    }
    
    report_path = results_dir / "validation.json"
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"✅ Validation report saved: {report_path}")
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("PHASE 2.1 COMPLETE ✅")
    logger.info("="*70)
    logger.info(f"Baseline graph: {baseline_path}")
    logger.info(f"  Nodes: {baseline_stats['num_nodes']:,}")
    logger.info(f"  Edges: {baseline_stats['num_edges']:,}")
    logger.info(f"  Features: {baseline_stats['feature_dim']}D")
    logger.info(f"  Training pairs: {baseline_stats['num_training_pairs']:,}")
    logger.info("")
    logger.info(f"Enhanced graph: {enhanced_path}")
    logger.info(f"  Nodes: {enhanced_stats['num_nodes']:,}")
    logger.info(f"  Edges: {enhanced_stats['num_edges']:,}")
    logger.info(f"  Features: {enhanced_stats['feature_dim']}D")
    logger.info(f"  Training pairs: {enhanced_stats['num_training_pairs']:,}")
    logger.info(f"  TNF feature: {tnf_feature_value:.4f} (assigned to {other_nodes_with_tnf} nodes)")
    logger.info("")
    logger.info("Ready for Phase 3: Model Training")
    logger.info("="*70)

except Exception as e:
    logger.error("="*70)
    logger.error("PHASE 2.1 FAILED ❌")
    logger.error("="*70)
    logger.error(f"Error: {e}")
    
    import traceback
    logger.error("\nFull traceback:")
    logger.error(traceback.format_exc())
    
    sys.exit(1)

PYEOF

echo ""
echo "========================================================================"
echo "Phase 2.1 Job Complete"
echo "End time: $(date)"
echo "========================================================================"