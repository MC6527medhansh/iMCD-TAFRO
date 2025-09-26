"""
Main entry point for iMCD-KG project
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def run_environment_test():
    """Run environment tests"""
    from scripts.environment_test import main as test_main
    return test_main()

def run_literature_summary():
    """Show literature summary"""
    from scripts.literature_summary import main as lit_main
    lit_main()

def run_kgml_replication():
    """Replicate KGML-xDTD baseline results"""
    from scripts.baselines.kgml_replication import main as kgml_main
    return kgml_main()

# def run_txgnn_baseline():
#     """Test TxGNN baseline"""
#     from scripts.baselines.txgnn_baseline import main as txgnn_main
#     return txgnn_main()

def run_feature_extraction():
    """Extract experimental features from paper"""
    from scripts.experimental.feature_extraction import main as extract_main
    return extract_main()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='iMCD-KG Project')
    parser.add_argument('command', 
                       choices=['test', 'literature', 'kgml', 'txgnn', 'extract', 'train'], 
                       help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        return run_environment_test()
    elif args.command == 'literature':
        run_literature_summary()
        return 0
    elif args.command == 'kgml':
        return run_kgml_replication()
    elif args.command == 'txgnn':
        return run_txgnn_baseline()
    elif args.command == 'extract':
        return run_feature_extraction()
    elif args.command == 'train':
        print("Training pipeline coming in Phase 3!")
        return 0

if __name__ == "__main__":
    exit(main())