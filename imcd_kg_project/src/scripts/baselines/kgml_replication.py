"""
KGML-xDTD baseline replication for iMCD-TAFRO
Goal: Verify adalimumab ranks #3 with score ~0.83736
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from config import IMCD_EXPERIMENTAL_DATA, KGML_REPO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_kgml_installation():
    """Check if KGML-xDTD is properly set up"""
    logger.info("Checking KGML-xDTD installation...")
    
    if not KGML_REPO.exists():
        logger.error(f"KGML-xDTD repository not found at {KGML_REPO}")
        logger.error("Run: cd src && git clone https://github.com/chunyuma/KGML-xDTD.git")
        return False
    
    logger.info("✅ KGML-xDTD repository found")
    return True

def download_data_instructions():
    """Print instructions for downloading KGML data"""
    print("\n" + "="*60)
    print("📥 REQUIRED DATA DOWNLOAD")
    print("="*60)
    
    print("\n1. Download RTX-KG2 data from Zenodo:")
    print("   wget https://zenodo.org/record/7582233/files/bkg_rtxkg2c_v2.7.3.tar.gz")
    print("   tar -xzf bkg_rtxkg2c_v2.7.3.tar.gz")
    
    print("\n2. Download training data:")
    print("   wget https://zenodo.org/record/7582233/files/training_data.tar.gz")
    print("   tar -xzf training_data.tar.gz")
    
    print("\n3. Download pretrained models:")
    print("   wget https://zenodo.org/record/7582233/files/models.tar.gz")
    print("   tar -xzf models.tar.gz")
    
    print(f"\n4. Total download size: ~28GB")
    print(f"   Place extracted files in: {KGML_REPO / 'data'}")

def test_baseline_prediction():
    """Show baseline target metrics"""
    target_data = IMCD_EXPERIMENTAL_DATA['kgml_results']
    
    print("\n" + "="*50)
    print("🎯 BASELINE TARGET METRICS")
    print("="*50)
    
    print(f"\nTarget Disease: iMCD-TAFRO")
    print(f"Expected adalimumab rank: #{target_data['adalimumab_rank']}")
    print(f"Expected adalimumab score: {target_data['adalimumab_score']}")
    print(f"Top drugs should be: {target_data['top_drugs']}")

def main():
    """Main function for KGML replication"""
    print("🕸️ KGML-xDTD Baseline Replication")
    print("="*50)
    
    if not check_kgml_installation():
        return 1
    
    download_data_instructions()
    test_baseline_prediction()
    
    print("\n✅ KGML baseline setup complete")
    print("💾 Download data to proceed with replication")
    
    return 0

if __name__ == "__main__":
    exit(main())