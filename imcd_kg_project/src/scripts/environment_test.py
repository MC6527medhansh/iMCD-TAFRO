#!/usr/bin/env python3
"""
Environment test script for iMCD-KG project
Run this to verify all dependencies are correctly installed
"""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_core_packages():
    """Test core data science packages"""
    logger.info("Testing core packages...")
    
    try:
        import pandas as pd
        import numpy as np
        logger.info(f"✅ Pandas {pd.__version__}")
        logger.info(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        logger.error(f"❌ Core packages failed: {e}")
        return False
    
    return True

def test_ml_packages():
    """Test machine learning packages"""
    logger.info("Testing ML packages...")
    
    try:
        import torch
        import torch_geometric
        import networkx as nx
        logger.info(f"✅ PyTorch {torch.__version__}")
        logger.info(f"✅ PyTorch Geometric {torch_geometric.__version__}")
        logger.info(f"✅ NetworkX {nx.__version__}")
    except ImportError as e:
        logger.error(f"❌ ML packages failed: {e}")
        return False
    
    return True

def test_bio_packages():
    """Test bioinformatics packages"""
    logger.info("Testing bioinformatics packages...")
    
    try:
        import scanpy as sc
        import anndata
        logger.info(f"✅ Scanpy {sc.__version__}")
        logger.info(f"✅ AnnData {anndata.__version__}")
    except ImportError as e:
        logger.error(f"❌ Bio packages failed: {e}")
        return False
    
    return True

def test_functionality():
    """Test basic functionality"""
    logger.info("Testing basic functionality...")
    
    try:
        import pandas as pd
        import torch
        
        # Test DataFrame creation
        df = pd.DataFrame({'test': [1, 2, 3]})
        assert len(df) == 3
        
        # Test PyTorch tensor creation
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.shape == (3,)
        
        logger.info("✅ Basic functionality working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Starting environment tests...")
    logger.info(f"Python version: {sys.version}")
    
    tests = [
        test_core_packages,
        test_ml_packages, 
        test_bio_packages,
        test_functionality
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    if all_passed:
        logger.info("🎉 All tests passed! Environment ready for iMCD-KG project.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check installation.")
        return 1

if __name__ == "__main__":
    exit(main())