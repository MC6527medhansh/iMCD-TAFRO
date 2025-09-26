"""
Extract experimental features from iMCD paper for graph enhancement
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
from config import IMCD_EXPERIMENTAL_DATA, EXPERIMENTAL_DATA_DIR
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_quantitative_features():
    """Convert paper findings to quantitative graph features"""
    logger.info("Extracting features from iMCD paper...")
    
    data = IMCD_EXPERIMENTAL_DATA
    
    features = {
        'gene_expression': {
            'TNF': {
                'log2_fold_change': data['scrna_seq']['naive_cd4_tnf_fold_change'],
                'linear_fold_change': 2 ** data['scrna_seq']['naive_cd4_tnf_fold_change'],
                'cell_type': 'naive_CD4_T_cell',
                'evidence_strength': 2.0
            }
        },
        'functional_validation': {
            'tnf_production_rate': data['functional_assays']['naive_cd4_tnf_producers']['imcd_percent'] / 100,
            'healthy_rate': data['functional_assays']['naive_cd4_tnf_producers']['healthy_percent'] / 100,
            'fold_difference': data['functional_assays']['naive_cd4_tnf_producers']['imcd_percent'] / 
                             data['functional_assays']['naive_cd4_tnf_producers']['healthy_percent'],
            'p_value': data['functional_assays']['naive_cd4_tnf_producers']['p_value']
        },
        'baseline_performance': {
            'adalimumab_current_rank': data['kgml_results']['adalimumab_rank'],
            'adalimumab_current_score': data['kgml_results']['adalimumab_score'],
            'target_rank': 1,
            'improvement_needed': data['kgml_results']['adalimumab_rank'] - 1
        }
    }
    
    return features

def save_features(features):
    """Save features to file"""
    EXPERIMENTAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(EXPERIMENTAL_DATA_DIR / 'quantitative_features.json', 'w') as f:
        json.dump(features, f, indent=2)
    
    logger.info(f"Features saved to {EXPERIMENTAL_DATA_DIR}")

def main():
    """Main function"""
    print("🔬 Experimental Feature Extraction")
    print("="*40)
    
    features = extract_quantitative_features()
    save_features(features)
    
    print(f"\n📊 EXTRACTED FEATURES:")
    print(f"• TNF fold change: {features['gene_expression']['TNF']['linear_fold_change']:.1f}x")
    print(f"• Current adalimumab rank: #{features['baseline_performance']['adalimumab_current_rank']}")
    print(f"• Target rank: #{features['baseline_performance']['target_rank']}")
    
    print(f"\n✅ Feature extraction complete")
    return 0

if __name__ == "__main__":
    exit(main())