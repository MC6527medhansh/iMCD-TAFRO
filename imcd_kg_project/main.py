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

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='iMCD-KG Project')
    parser.add_argument('command', choices=['test', 'literature', 'analyze', 'train'], 
                       help='Command to run')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        return run_environment_test()
    elif args.command == 'literature':
        run_literature_summary()
        return 0
    elif args.command == 'analyze':
        print("🔬 Analysis pipeline coming in Day 2-3!")
        return 0
    elif args.command == 'train':
        print("🤖 Training pipeline coming in Day 14+!")
        return 0

if __name__ == "__main__":
    exit(main())