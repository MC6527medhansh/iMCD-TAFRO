#!/usr/bin/env python3
"""
Comprehensive Phase 2.1 Testing Suite
Tests EVERYTHING before submitting to Sockeye

Run this to validate:
1. Config paths correct
2. Enhanced predictor works with full graph
3. Baseline graph builds correctly
4. Enhanced graph builds correctly
5. All validation checks pass
6. No data leakage
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import logging
import torch
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Phase2Tester:
    """Comprehensive testing for Phase 2.1"""
    
    def __init__(self):
        self.results = {}
        
    def test_1_config_import(self):
        """TEST 1: Can we import config with correct paths?"""
        logger.info("="*60)
        logger.info("TEST 1: Config Import")
        logger.info("="*60)
        
        try:
            from config import (
                PROCESSED_GRAPH_PATH,
                ADALIMUMAB_ID,
                CASTLEMAN_ID,
                TNF_ID,
                TNF_LOG2_FOLD_CHANGE,
                DATA_DIR
            )
            
            logger.info(f"✅ Config imported")
            logger.info(f"   Graph path: {PROCESSED_GRAPH_PATH}")
            logger.info(f"   Adalimumab: {ADALIMUMAB_ID}")
            logger.info(f"   Castleman: {CASTLEMAN_ID}")
            logger.info(f"   TNF: {TNF_ID}")
            logger.info(f"   TNF fold change: {TNF_LOG2_FOLD_CHANGE}")
            
            # Check graph exists
            if PROCESSED_GRAPH_PATH.exists():
                size_gb = PROCESSED_GRAPH_PATH.stat().st_size / 1e9
                logger.info(f"✅ Graph file exists: {size_gb:.2f} GB")
                self.results['test_1'] = {'status': 'PASS', 'graph_size_gb': size_gb}
                return True
            else:
                logger.error(f"❌ Graph file NOT found: {PROCESSED_GRAPH_PATH}")
                self.results['test_1'] = {'status': 'FAIL', 'reason': 'graph_missing'}
                return False
                
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            self.results['test_1'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_2_predictor_initialization(self):
        """TEST 2: Can we initialize the predictor?"""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Predictor Initialization")
        logger.info("="*60)
        
        try:
            from enhanced_kgnn.enhanced_predictor import ExperimentalGraphPredictor
            
            predictor = ExperimentalGraphPredictor()
            
            logger.info("✅ Predictor initialized")
            logger.info(f"   Adalimumab ID: {predictor.adalimumab_id}")
            logger.info(f"   Castleman ID: {predictor.castleman_id}")
            logger.info(f"   TNF ID: {predictor.tnf_id}")
            logger.info(f"   TNF fold change: {predictor.tnf_fold_change:.2f}x")
            
            self.results['test_2'] = {'status': 'PASS'}
            return predictor
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            self.results['test_2'] = {'status': 'FAIL', 'error': str(e)}
            return None
    
    def test_3_training_data_loading(self, predictor):
        """TEST 3: Can we load training data without errors?"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: Training Data Loading")
        logger.info("="*60)
        
        try:
            training_pairs = predictor.load_training_data()
            
            num_pairs = len(training_pairs)
            num_positive = sum(1 for _, _, label in training_pairs if label == 1)
            num_negative = sum(1 for _, _, label in training_pairs if label == 0)
            
            logger.info(f"✅ Training data loaded")
            logger.info(f"   Total pairs: {num_pairs:,}")
            logger.info(f"   Positive: {num_positive:,}")
            logger.info(f"   Negative: {num_negative:,}")
            
            # Check for data leakage (should be caught by load_training_data)
            adalimumab_castleman = [
                (d, dis) for d, dis, _ in training_pairs 
                if d == predictor.adalimumab_id and dis == predictor.castleman_id
            ]
            
            if len(adalimumab_castleman) == 0:
                logger.info(f"✅ No data leakage detected")
                self.results['test_3'] = {
                    'status': 'PASS',
                    'num_pairs': num_pairs,
                    'no_leakage': True
                }
                return True
            else:
                logger.error(f"❌ DATA LEAKAGE: {len(adalimumab_castleman)} adalimumab-Castleman pairs found!")
                self.results['test_3'] = {
                    'status': 'FAIL',
                    'reason': 'data_leakage',
                    'leakage_count': len(adalimumab_castleman)
                }
                return False
                
        except ValueError as e:
            # This is expected if data leakage is found (predictor raises ValueError)
            if "Data leakage" in str(e):
                logger.error(f"❌ Data leakage detected (as expected error): {e}")
                self.results['test_3'] = {'status': 'FAIL', 'reason': 'data_leakage_detected'}
                return False
            else:
                raise
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            self.results['test_3'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_4_baseline_graph_construction(self, predictor):
        """TEST 4: Can we build baseline graph (3D)?"""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: Baseline Graph Construction (3D)")
        logger.info("="*60)
        
        try:
            torch.manual_seed(42)
            np.random.seed(42)
            
            baseline_data = predictor.build_graph_with_experimental_features(use_experimental=False)
            
            # Validate structure
            assert baseline_data.x.shape[1] == 3, f"Wrong feature dim: {baseline_data.x.shape[1]}"
            assert baseline_data.num_nodes > 0, "No nodes"
            assert baseline_data.edge_index.shape[1] > 0, "No edges"
            assert baseline_data.train_edge_labels.shape[0] > 0, "No training labels"
            
            # Validate no NaN/Inf
            assert not torch.isnan(baseline_data.x).any(), "NaN in features"
            assert not torch.isinf(baseline_data.x).any(), "Inf in features"
            
            # Validate features are valid
            assert (baseline_data.x >= 0).all(), "Negative features"
            assert (baseline_data.x <= 1).all(), "Features > 1"
            
            # Validate edge indices
            assert baseline_data.edge_index.max() < baseline_data.num_nodes, "Edge index out of bounds"
            
            logger.info(f"✅ Baseline graph validated")
            logger.info(f"   Nodes: {baseline_data.num_nodes:,}")
            logger.info(f"   Edges: {baseline_data.edge_index.shape[1]:,}")
            logger.info(f"   Features: {baseline_data.x.shape[1]}D")
            logger.info(f"   Training pairs: {baseline_data.train_edge_labels.shape[0]:,}")
            
            self.results['test_4'] = {
                'status': 'PASS',
                'num_nodes': baseline_data.num_nodes,
                'num_edges': baseline_data.edge_index.shape[1],
                'feature_dim': baseline_data.x.shape[1]
            }
            
            return baseline_data
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            self.results['test_4'] = {'status': 'FAIL', 'error': str(e)}
            return None
    
    def test_5_enhanced_graph_construction(self, predictor):
        """TEST 5: Can we build enhanced graph (4D with TNF)?"""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: Enhanced Graph Construction (4D + TNF)")
        logger.info("="*60)
        
        try:
            torch.manual_seed(42)
            np.random.seed(42)
            
            enhanced_data = predictor.build_graph_with_experimental_features(use_experimental=True)
            
            # Validate structure
            assert enhanced_data.x.shape[1] == 4, f"Wrong feature dim: {enhanced_data.x.shape[1]}"
            assert enhanced_data.num_nodes > 0, "No nodes"
            assert enhanced_data.edge_index.shape[1] > 0, "No edges"
            
            # Validate TNF feature
            tnf_idx = predictor.entity_to_idx[predictor.tnf_id]
            tnf_feature = enhanced_data.x[tnf_idx, 3].item()
            
            assert tnf_feature > 0, f"TNF feature not assigned: {tnf_feature}"
            assert abs(tnf_feature - 4.94) < 0.01, f"TNF feature wrong: {tnf_feature}"
            
            # Count nodes with TNF feature
            num_tnf_nodes = (enhanced_data.x[:, 3] > 0).sum().item()
            
            logger.info(f"✅ Enhanced graph validated")
            logger.info(f"   Nodes: {enhanced_data.num_nodes:,}")
            logger.info(f"   Edges: {enhanced_data.edge_index.shape[1]:,}")
            logger.info(f"   Features: {enhanced_data.x.shape[1]}D")
            logger.info(f"   TNF feature: {tnf_feature:.4f}")
            logger.info(f"   Nodes with TNF feature: {num_tnf_nodes}")
            
            self.results['test_5'] = {
                'status': 'PASS',
                'num_nodes': enhanced_data.num_nodes,
                'num_edges': enhanced_data.edge_index.shape[1],
                'feature_dim': enhanced_data.x.shape[1],
                'tnf_feature_value': tnf_feature,
                'num_tnf_nodes': num_tnf_nodes
            }
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            self.results['test_5'] = {'status': 'FAIL', 'error': str(e)}
            return None
    
    def test_6_graph_comparison(self, baseline_data, enhanced_data):
        """TEST 6: Are baseline and enhanced graphs consistent?"""
        logger.info("\n" + "="*60)
        logger.info("TEST 6: Baseline vs Enhanced Comparison")
        logger.info("="*60)
        
        try:
            # Same number of nodes
            assert baseline_data.num_nodes == enhanced_data.num_nodes, "Node count mismatch"
            logger.info(f"✅ Same number of nodes: {baseline_data.num_nodes:,}")
            
            # Same edges
            assert baseline_data.edge_index.shape[1] == enhanced_data.edge_index.shape[1], "Edge count mismatch"
            assert torch.equal(baseline_data.edge_index, enhanced_data.edge_index), "Edge structure different"
            logger.info(f"✅ Same edge structure: {baseline_data.edge_index.shape[1]:,} edges")
            
            # First 3 dimensions should match
            first_3d_match = torch.allclose(baseline_data.x, enhanced_data.x[:, :3])
            assert first_3d_match, "First 3D features don't match"
            logger.info(f"✅ First 3D features match")
            
            # 4th dimension should be mostly zeros (except TNF nodes)
            fourth_dim = enhanced_data.x[:, 3]
            num_nonzero = (fourth_dim > 0).sum().item()
            logger.info(f"✅ 4th dimension: {num_nonzero} non-zero values")
            
            self.results['test_6'] = {
                'status': 'PASS',
                'same_structure': True,
                'first_3d_match': True,
                'num_tnf_features': num_nonzero
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            self.results['test_6'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_7_model_initialization(self, baseline_data, enhanced_data):
        """TEST 7: Can we initialize models with correct dimensions?"""
        logger.info("\n" + "="*60)
        logger.info("TEST 7: Model Initialization")
        logger.info("="*60)
        
        try:
            from enhanced_kgnn.enhanced_predictor import GraphSAGEModel
            
            # Baseline model (3D input)
            baseline_model = GraphSAGEModel(input_dim=3, hidden_dim=64, output_dim=32)
            logger.info(f"✅ Baseline model initialized (3D input)")
            
            # Enhanced model (4D input)
            enhanced_model = GraphSAGEModel(input_dim=4, hidden_dim=64, output_dim=32)
            logger.info(f"✅ Enhanced model initialized (4D input)")
            
            # Test forward pass
            with torch.no_grad():
                z_baseline = baseline_model(baseline_data.x, baseline_data.edge_index)
                z_enhanced = enhanced_model(enhanced_data.x, enhanced_data.edge_index)
            
            assert z_baseline.shape == (baseline_data.num_nodes, 32), "Wrong output shape"
            assert z_enhanced.shape == (enhanced_data.num_nodes, 32), "Wrong output shape"
            
            logger.info(f"✅ Forward pass successful")
            logger.info(f"   Baseline output: {z_baseline.shape}")
            logger.info(f"   Enhanced output: {z_enhanced.shape}")
            
            self.results['test_7'] = {
                'status': 'PASS',
                'baseline_output_shape': list(z_baseline.shape),
                'enhanced_output_shape': list(z_enhanced.shape)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            self.results['test_7'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE PHASE 2.1 TESTING SUITE")
        logger.info("="*70 + "\n")
        
        # Test 1: Config
        if not self.test_1_config_import():
            logger.error("❌ Config import failed - cannot continue")
            return False
        
        # Test 2: Predictor initialization
        predictor = self.test_2_predictor_initialization()
        if predictor is None:
            logger.error("❌ Predictor initialization failed - cannot continue")
            return False
        
        # Test 3: Training data
        if not self.test_3_training_data_loading(predictor):
            logger.error("❌ Training data loading failed - cannot continue")
            return False
        
        # Test 4: Baseline graph
        baseline_data = self.test_4_baseline_graph_construction(predictor)
        if baseline_data is None:
            logger.error("❌ Baseline graph construction failed - cannot continue")
            return False
        
        # Test 5: Enhanced graph
        enhanced_data = self.test_5_enhanced_graph_construction(predictor)
        if enhanced_data is None:
            logger.error("❌ Enhanced graph construction failed - cannot continue")
            return False
        
        # Test 6: Comparison
        if not self.test_6_graph_comparison(baseline_data, enhanced_data):
            logger.error("❌ Graph comparison failed")
            return False
        
        # Test 7: Model initialization
        if not self.test_7_model_initialization(baseline_data, enhanced_data):
            logger.error("❌ Model initialization failed")
            return False
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("TEST SUMMARY")
        logger.info("="*70)
        
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        
        logger.info(f"Passed: {passed}/7")
        logger.info(f"Failed: {failed}/7")
        
        for test_name, result in self.results.items():
            status = "✅" if result['status'] == 'PASS' else "❌"
            logger.info(f"{status} {test_name}: {result['status']}")
        
        if failed == 0:
            logger.info("\n" + "="*70)
            logger.info("🎉 ALL TESTS PASSED - READY FOR SOCKEYE")
            logger.info("="*70)
            logger.info("\nNext steps:")
            logger.info("1. git add src/enhanced_kgnn/enhanced_predictor.py jobs/phase_2_1.sh")
            logger.info("2. git commit -m 'Phase 2.1: Build baseline + enhanced graphs'")
            logger.info("3. git push")
            logger.info("4. On Sockeye: git pull && sbatch jobs/phase_2_1.sh")
            return True
        else:
            logger.error("\n" + "="*70)
            logger.error("❌ SOME TESTS FAILED - FIX BEFORE SUBMITTING")
            logger.error("="*70)
            return False


def main():
    """Run comprehensive tests"""
    tester = Phase2Tester()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())