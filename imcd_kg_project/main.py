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

def run_scrna_analysis():
    """Run single-cell RNA-seq analysis"""
    from scripts.preprocessing.scrna_analysis import main as scrna_main
    return scrna_main()

def run_imcd_analysis():
    """Run iMCD experimental data analysis"""
    from scripts.analysis.imcd_data_analysis import main as imcd_main
    return imcd_main()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='iMCD-KG Project')
    parser.add_argument('command', 
                       choices=['test', 'literature', 'scrna', 'imcd', 'kgml', 'train'], 
                       help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        return run_environment_test()
    elif args.command == 'literature':
        run_literature_summary()
        return 0
    elif args.command == 'scrna':
        return run_scrna_analysis()
    elif args.command == 'imcd':
        return run_imcd_analysis()
    elif args.command == 'kgml':
        print("🕸️ KGML-xDTD replication coming in Day 6!")
        return 0
    elif args.command == 'train':
        print("🤖 Training pipeline coming in Day 14+!")
        return 0

if __name__ == "__main__":
    exit(main())