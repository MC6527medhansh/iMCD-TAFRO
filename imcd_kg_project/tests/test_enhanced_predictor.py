#!/usr/bin/env python3
"""
Incremental Testing Suite for Enhanced Predictor
Location: tests/test_enhanced_predictor.py
Strategy: Test each component independently before integration
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root (tests/ -> imcd_kg_project/)
PROJECT_ROOT = Path(__file__).parent.parent

# Add src/enhanced_kgnn to path
sys.path.insert(0, str(PROJECT_ROOT / "src" / "enhanced_kgnn"))

from rtx_kg_loader import RTXKGLoader
from enhanced_predictor import ExperimentalGraphPredictor, GraphSAGEModel

class TestEnhancedPredictor:
    """Incremental testing following Google's principles"""
    
    def __init__(self):
        # All paths relative to project root
        self.kg_path = PROJECT_ROOT / "data" / "kgml_data" / "bkg_rtxkg2c_v2.7.3"
        self.training_path = PROJECT_ROOT / "data" / "kgml_data" / "training_data"
        self.results = {}
        
        # Validate paths exist
        if not self.kg_path.exists():
            raise FileNotFoundError(f"KG data not found: {self.kg_path}")
        if not self.training_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.training_path}")
    
    def test_1_data_loading(self):
        """TEST 1: Can we load training data without errors?"""
        logger.info("="*60)
        logger.info("TEST 1: Data Loading")
        logger.info("="*60)
        
        try:
            loader = RTXKGLoader(self.kg_path)
            predictor = ExperimentalGraphPredictor(loader)
            
            # Test training data loading
            pairs = predictor.load_training_data()
            
            # Validate data quality
            assert len(pairs) > 0, "No training pairs loaded"
            
            # Check data types
            for i, (drug, disease, label) in enumerate(pairs[:10]):
                assert isinstance(drug, str), f"Row {i}: Drug is not string: {type(drug)}"
                assert isinstance(disease, str), f"Row {i}: Disease is not string: {type(disease)}"
                assert label in [0, 1], f"Row {i}: Invalid label: {label}"
                
            logger.info(f"✅ PASS: Loaded {len(pairs)} valid pairs")
            logger.info(f"   Sample: {pairs[0]}")
            
            self.results['test_1'] = {
                'status': 'PASS',
                'num_pairs': len(pairs),
                'sample': pairs[0]
            }
            return predictor
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            self.results['test_1'] = {'status': 'FAIL', 'error': str(e)}
            raise
    
    def test_2_entity_mapping(self, predictor):
        """TEST 2: Does entity mapping handle mixed types correctly?"""
        logger.info("="*60)
        logger.info("TEST 2: Entity Mapping (Bug Fix Validation)")
        logger.info("="*60)
        
        try:
            pairs = predictor.load_training_data()
            
            # Extract entities and check for type issues
            entities = set()
            for drug, disease, _ in pairs:
                entities.add(drug)
                entities.add(disease)
            
            # This is where the previous bug occurred
            entity_list = sorted(entities)
            
            # Validate all are strings
            non_strings = [e for e in entity_list if not isinstance(e, str)]
            assert len(non_strings) == 0, f"Found non-string entities: {non_strings[:5]}"
            
            # Check for None or NaN
            invalid = [e for e in entity_list if e is None or (isinstance(e, float) and np.isnan(e))]
            assert len(invalid) == 0, f"Found invalid entities: {invalid[:5]}"
            
            logger.info(f"✅ PASS: All {len(entity_list)} entities are valid strings")
            logger.info(f"   Drugs: {sum(1 for e in entity_list if e.startswith('CHEMBL'))}")
            logger.info(f"   Diseases: {sum(1 for e in entity_list if e.startswith('MONDO'))}")
            
            self.results['test_2'] = {
                'status': 'PASS',
                'num_entities': len(entity_list),
                'num_drugs': sum(1 for e in entity_list if e.startswith('CHEMBL')),
                'num_diseases': sum(1 for e in entity_list if e.startswith('MONDO'))
            }
            return entity_list
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            self.results['test_2'] = {'status': 'FAIL', 'error': str(e)}
            raise
    
    def test_3_baseline_graph(self, predictor):
        """TEST 3: Can we build baseline graph with correct dimensions?"""
        logger.info("="*60)
        logger.info("TEST 3: Baseline Graph Construction (3D features)")
        logger.info("="*60)
        
        try:
            # Build WITHOUT experimental features
            data = predictor.build_graph_with_experimental_features(use_experimental=False)
            
            # Validate tensor dimensions
            num_nodes = data.x.shape[0]
            num_features = data.x.shape[1]
            num_edges = data.edge_index.shape[1]
            
            assert num_features == 3, f"Expected 3D features, got {num_features}D"
            assert num_nodes > 0, "No nodes in graph"
            assert num_edges > 0, "No edges in graph"
            
            # Check feature values are valid
            assert torch.all(torch.isfinite(data.x)), "Found NaN/Inf in features"
            assert torch.all(data.x >= 0), "Found negative values in features"
            
            logger.info(f"✅ PASS: Baseline graph constructed")
            logger.info(f"   Nodes: {num_nodes}")
            logger.info(f"   Edges: {num_edges}")
            logger.info(f"   Features: {num_features}D")
            logger.info(f"   Feature range: [{data.x.min():.2f}, {data.x.max():.2f}]")
            
            self.results['test_3'] = {
                'status': 'PASS',
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'num_features': num_features
            }
            return data
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            self.results['test_3'] = {'status': 'FAIL', 'error': str(e)}
            raise
    
    def test_4_enhanced_graph(self, predictor):
        """TEST 4: Can we build enhanced graph with experimental features?"""
        logger.info("="*60)
        logger.info("TEST 4: Enhanced Graph Construction (4D features)")
        logger.info("="*60)
        
        try:
            # Build WITH experimental features
            data = predictor.build_graph_with_experimental_features(use_experimental=True)
            
            # Validate tensor dimensions
            num_nodes = data.x.shape[0]
            num_features = data.x.shape[1]
            num_edges = data.edge_index.shape[1]
            
            assert num_features == 4, f"Expected 4D features, got {num_features}D"
            assert num_nodes > 0, "No nodes in graph"
            assert num_edges > 0, "No edges in graph"
            
            # Check experimental features were added
            experimental_col = data.x[:, 3]
            num_with_features = torch.sum(experimental_col > 0).item()
            
            assert num_with_features > 0, "No experimental features added!"
            
            logger.info(f"✅ PASS: Enhanced graph constructed")
            logger.info(f"   Nodes: {num_nodes}")
            logger.info(f"   Edges: {num_edges}")
            logger.info(f"   Features: {num_features}D")
            logger.info(f"   Nodes with experimental features: {num_with_features}")
            logger.info(f"   Experimental feature range: [{experimental_col.min():.2f}, {experimental_col.max():.2f}]")
            
            self.results['test_4'] = {
                'status': 'PASS',
                'num_nodes': num_nodes,
                'num_features': num_features,
                'num_with_exp_features': num_with_features
            }
            return data
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            self.results['test_4'] = {'status': 'FAIL', 'error': str(e)}
            raise
    
    def test_5_model_initialization(self, baseline_data, enhanced_data):
        """TEST 5: Can models initialize with correct dimensions?"""
        logger.info("="*60)
        logger.info("TEST 5: Model Initialization")
        logger.info("="*60)
        
        try:
            # Test baseline model (3D input)
            baseline_model = GraphSAGEModel(input_dim=3, hidden_dim=64, output_dim=32)
            
            # Test forward pass with baseline data
            with torch.no_grad():
                z_baseline = baseline_model(baseline_data.x, baseline_data.edge_index)
            
            assert z_baseline.shape[0] == baseline_data.x.shape[0], "Node count mismatch"
            assert z_baseline.shape[1] == 32, "Output dimension mismatch"
            
            logger.info(f"✅ PASS: Baseline model initialized")
            logger.info(f"   Input: {baseline_data.x.shape}")
            logger.info(f"   Output: {z_baseline.shape}")
            
            # Test enhanced model (4D input)
            enhanced_model = GraphSAGEModel(input_dim=4, hidden_dim=64, output_dim=32)
            
            # Test forward pass with enhanced data
            with torch.no_grad():
                z_enhanced = enhanced_model(enhanced_data.x, enhanced_data.edge_index)
            
            assert z_enhanced.shape[0] == enhanced_data.x.shape[0], "Node count mismatch"
            assert z_enhanced.shape[1] == 32, "Output dimension mismatch"
            
            logger.info(f"✅ PASS: Enhanced model initialized")
            logger.info(f"   Input: {enhanced_data.x.shape}")
            logger.info(f"   Output: {z_enhanced.shape}")
            
            self.results['test_5'] = {
                'status': 'PASS',
                'baseline_output_shape': list(z_baseline.shape),
                'enhanced_output_shape': list(z_enhanced.shape)
            }
            return baseline_model, enhanced_model
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            self.results['test_5'] = {'status': 'FAIL', 'error': str(e)}
            raise
    
    def test_6_target_entities_exist(self, predictor):
        """TEST 6: Are our target entities (adalimumab, Castleman) in the graph?"""
        logger.info("="*60)
        logger.info("TEST 6: Target Entity Validation")
        logger.info("="*60)
        
        try:
            # Check if target entities exist
            adalimumab_exists = predictor.adalimumab_id in predictor.entity_to_idx
            castleman_exists = predictor.castleman_id in predictor.entity_to_idx
            
            logger.info(f"   Adalimumab ({predictor.adalimumab_id}): {'✅ FOUND' if adalimumab_exists else '❌ MISSING'}")
            logger.info(f"   Castleman ({predictor.castleman_id}): {'✅ FOUND' if castleman_exists else '❌ MISSING'}")
            
            if not adalimumab_exists:
                logger.warning("Adalimumab not in training data - cannot evaluate ranking!")
            if not castleman_exists:
                logger.warning("Castleman disease not in training data - cannot evaluate ranking!")
            
            self.results['test_6'] = {
                'status': 'PASS' if (adalimumab_exists and castleman_exists) else 'WARNING',
                'adalimumab_exists': adalimumab_exists,
                'castleman_exists': castleman_exists
            }
            
            return adalimumab_exists and castleman_exists
            
        except Exception as e:
            logger.error(f"❌ FAIL: {e}")
            self.results['test_6'] = {'status': 'FAIL', 'error': str(e)}
            raise
    
    def print_summary(self):
        """Print test summary"""
        logger.info("="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        warnings = sum(1 for r in self.results.values() if r['status'] == 'WARNING')
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⚠️  Warnings: {warnings}")
        logger.info("")
        
        for test_name, result in self.results.items():
            status_symbol = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⚠️"
            logger.info(f"{status_symbol} {test_name}: {result['status']}")
        
        return passed == total

def main():
    """Run all tests incrementally"""
    print("🧪 INCREMENTAL TESTING SUITE")
    print("Following Google's principle: Test at every step")
    print("")
    
    tester = TestEnhancedPredictor()
    
    try:
        # Test 1: Data Loading
        predictor = tester.test_1_data_loading()
        
        # Test 2: Entity Mapping (bug fix validation)
        entity_list = tester.test_2_entity_mapping(predictor)
        
        # Test 3: Baseline Graph (3D)
        baseline_data = tester.test_3_baseline_graph(predictor)
        
        # Test 4: Enhanced Graph (4D)
        enhanced_data = tester.test_4_enhanced_graph(predictor)
        
        # Test 5: Model Initialization
        baseline_model, enhanced_model = tester.test_5_model_initialization(baseline_data, enhanced_data)
        
        # Test 6: Target Entities
        targets_exist = tester.test_6_target_entities_exist(predictor)
        
        # Print summary
        all_passed = tester.print_summary()
        
        if all_passed:
            print("\n🎉 ALL TESTS PASSED! Ready for training.")
            return 0
        else:
            print("\n⚠️  Some tests failed. Review errors above.")
            return 1
            
    except Exception as e:
        logger.error(f"\n💥 Testing aborted: {e}")
        tester.print_summary()
        return 1

if __name__ == "__main__":
    exit(main())