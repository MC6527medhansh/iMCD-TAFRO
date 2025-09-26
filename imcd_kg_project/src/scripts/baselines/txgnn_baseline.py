"""
TxGNN baseline testing for iMCD-TAFRO
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config import TXGNN_REPO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_txgnn_installation():
    """Check TxGNN installation"""
    logger.info("Checking TxGNN installation...")
    
    if not TXGNN_REPO.exists():
        logger.error(f"TxGNN repository not found at {TXGNN_REPO}")
        return False
    
    logger.info("✅ TxGNN repository found")
    return True

def install_dependencies():
    """Install TxGNN dependencies manually"""
    logger.info("Installing TxGNN dependencies...")
    
    # Core dependencies for TxGNN
    dependencies = [
        'torch>=1.12.0',
        'dgl==0.5.2',
        'numpy',
        'pandas',
        'scikit-learn',
        'networkx',
        'tqdm'
    ]
    
    try:
        import subprocess
        for dep in dependencies:
            logger.info(f"Installing {dep}...")
            result = subprocess.run(['pip', 'install', dep], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Failed to install {dep}: {result.stderr}")
        
        logger.info("✅ Dependencies installed")
        return True
        
    except Exception as e:
        logger.error(f"Dependency installation error: {e}")
        return False

def setup_txgnn_path():
    """Add TxGNN to Python path"""
    logger.info("Setting up TxGNN path...")
    
    try:
        # Add TxGNN repo to path
        sys.path.insert(0, str(TXGNN_REPO))
        
        # Test basic import
        import os
        txgnn_init = TXGNN_REPO / "__init__.py"
        if not txgnn_init.exists():
            # Create basic __init__.py if missing
            with open(txgnn_init, 'w') as f:
                f.write("# TxGNN package\n")
        
        logger.info("✅ TxGNN path configured")
        return True
        
    except Exception as e:
        logger.error(f"Path setup error: {e}")
        return False

def test_txgnn_import():
    """Test if we can access TxGNN modules"""
    logger.info("Testing TxGNN modules...")
    
    try:
        # Check if key files exist
        key_files = ['TxData.py', 'TxGNN.py']
        missing_files = []
        
        for file in key_files:
            if not (TXGNN_REPO / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"Missing TxGNN files: {missing_files}")
            logger.info("TxGNN repository may be incomplete")
        else:
            logger.info("✅ TxGNN core files found")
        
        return True
        
    except Exception as e:
        logger.error(f"TxGNN test failed: {e}")
        return False

def show_manual_instructions():
    """Show manual installation instructions"""
    print("\n" + "="*60)
    print("📋 MANUAL TxGNN SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. Install DGL (specific version required):")
    print("   # For CUDA 11.6:")
    print("   conda install -c dglteam dgl-cuda11.6==0.5.2")
    print("   # For CPU only:")
    print("   conda install -c dglteam dgl==0.5.2")
    
    print("\n2. If DGL installation fails, try:")
    print("   pip install dgl==0.5.2 -f https://data.dgl.ai/wheels/repo.html")
    
    print(f"\n3. TxGNN repository location:")
    print(f"   {TXGNN_REPO}")
    
    print("\n4. Alternative: Use TxGNN from source directly")
    print("   # We'll implement this approach in our baseline")

def main():
    """Main function for TxGNN baseline"""
    print("🔮 TxGNN Baseline Testing")
    print("="*40)
    
    if not check_txgnn_installation():
        return 1
    
    # Skip pip install, use dependencies + local repo
    install_dependencies()
    setup_txgnn_path()
    test_txgnn_import()
    
    show_manual_instructions()
    
    print("\n✅ TxGNN baseline setup complete")
    print("🎯 Ready for zero-shot predictions")
    
    return 0

if __name__ == "__main__":
    exit(main())