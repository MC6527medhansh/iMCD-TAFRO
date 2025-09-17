"""
Analysis of iMCD-TAFRO experimental data from the paper
Extracting key insights for graph enhancement
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import IMCD_EXPERIMENTAL_DATA, RESULTS_DIR
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class iMCDExperimentalDataAnalyzer:
    """
    Analyzer for experimental data from iMCD-TAFRO paper
    Prepares features for knowledge graph enhancement
    """
    
    def __init__(self):
        self.data = IMCD_EXPERIMENTAL_DATA
        self.results_dir = RESULTS_DIR / "imcd_analysis" 
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_proteomics_data(self):
        """Analyze proteomics findings"""
        logger.info("Analyzing proteomics data...")
        
        proteomics = self.data['proteomics']
        
        # Key pathway enrichments from paper
        enriched_pathways = {
            'TNF-via-NF-κB': {'enrichment': 'high', 'relevance': 'primary_target'},
            'IL-6-JAK-STAT3': {'enrichment': 'high', 'relevance': 'established_target'},
            'inflammatory_response': {'enrichment': 'high', 'relevance': 'broad_inflammation'},
            'interferon_gamma_response': {'enrichment': 'moderate', 'relevance': 'immune_activation'},
            'IL-2-STAT5': {'enrichment': 'moderate', 'relevance': 'tcell_activation'}
        }
        
        # Convert to feature scores for graph
        pathway_scores = {}
        for pathway, info in enriched_pathways.items():
            if info['enrichment'] == 'high':
                pathway_scores[pathway] = 2.0
            elif info['enrichment'] == 'moderate':
                pathway_scores[pathway] = 1.0
            else:
                pathway_scores[pathway] = 0.0
        
        logger.info("Pathway enrichment scores:")
        for pathway, score in pathway_scores.items():
            print(f"  {pathway}: {score}")
            
        return pathway_scores
    
    def analyze_scrna_data(self):
        """Analyze single-cell RNA-seq findings"""
        logger.info("Analyzing scRNA-seq data...")
        
        scrna = self.data['scrna_seq']
        
        # Key finding: TNF expression in naive CD4+ T cells
        cell_type_features = {
            'naive_CD4_T_cell': {
                'TNF_log2_fold_change': scrna['naive_cd4_tnf_fold_change'],
                'TNF_linear_fold_change': 2 ** scrna['naive_cd4_tnf_fold_change'],
                'disease_relevance': 'primary',
                'therapeutic_target': True
            },
            'CD4_T_cell_general': {
                'TNF_expression': 'elevated',
                'disease_relevance': 'high',
                'therapeutic_target': True
            }
        }
        
        logger.info("Cell type features:")
        for cell_type, features in cell_type_features.items():
            print(f"  {cell_type}:")
            for feature, value in features.items():
                print(f"    {feature}: {value}")
        
        return cell_type_features
    
    def analyze_functional_data(self):
        """Analyze functional assay results"""
        logger.info("Analyzing functional assay data...")
        
        functional = self.data['functional_assays']
        
        # TNF production rates
        tnf_production = functional['naive_cd4_tnf_producers']
        
        # Calculate fold difference and statistical significance
        fold_difference = tnf_production['imcd_percent'] / tnf_production['healthy_percent']
        
        functional_features = {
            'naive_CD4_TNF_production': {
                'imcd_rate': tnf_production['imcd_percent'] / 100,  # Convert to fraction
                'healthy_rate': tnf_production['healthy_percent'] / 100,
                'fold_difference': fold_difference,
                'p_value': tnf_production['p_value'],
                'significance': 'high' if tnf_production['p_value'] < 0.01 else 'moderate'
            }
        }
        
        logger.info("Functional assay features:")
        for assay, features in functional_features.items():
            print(f"  {assay}:")
            for feature, value in features.items():
                print(f"    {feature}: {value}")
        
        return functional_features
    
    def analyze_kgml_results(self):
        """Analyze KGML-xDTD results"""
        logger.info("Analyzing KGML-xDTD baseline results...")
        
        kgml = self.data['kgml_results']
        
        baseline_results = {
            'adalimumab': {
                'current_rank': kgml['adalimumab_rank'],
                'current_score': kgml['adalimumab_score'],
                'target_rank': 1,  # Our goal
                'improvement_needed': kgml['adalimumab_rank'] - 1
            },
            'top_drugs': kgml['top_drugs']
        }
        
        logger.info("Baseline KGML results:")
        print(f"  Current adalimumab rank: #{baseline_results['adalimumab']['current_rank']}")
        print(f"  Current adalimumab score: {baseline_results['adalimumab']['current_score']}")
        print(f"  Target rank: #{baseline_results['adalimumab']['target_rank']}")
        print(f"  Improvement needed: +{baseline_results['adalimumab']['improvement_needed']} positions")
        
        return baseline_results
    
    def create_graph_features(self):
        """Create features for graph neural network enhancement"""
        logger.info("Creating graph enhancement features...")
        
        # Combine all experimental evidence
        graph_features = {
            'disease_features': {
                'iMCD-TAFRO': {
                    'TNF_pathway_evidence': 2.0,  # Strong evidence
                    'IL6_pathway_evidence': 2.0,  # Strong evidence
                    'CD4_T_cell_involvement': 2.0,  # Strong evidence
                    'rare_disease': True,
                    'mortality_risk': 'high'
                }
            },
            'drug_features': {
                'adalimumab': {
                    'TNF_inhibitor': True,
                    'experimental_evidence_score': 2.0,  # Based on all evidence
                    'mechanism_match': 'direct',
                    'clinical_evidence': True  # N=1 success
                },
                'tocilizumab': {
                    'IL6_inhibitor': True,
                    'current_standard': True,
                    'limited_efficacy': True
                },
                'siltuximab': {
                    'IL6_inhibitor': True,
                    'FDA_approved': True,
                    'limited_efficacy': True
                }
            },
            'gene_features': {
                'TNF': {
                    'fold_change_naive_cd4': 4.94,
                    'functional_validation': True,
                    'pathway_centrality': 'high',
                    'therapeutic_target': True
                },
                'IL6': {
                    'pathway_enrichment': 'high',
                    'established_target': True
                }
            },
            'cell_type_features': {
                'naive_CD4_T_cell': {
                    'TNF_production_fold': 2.53,  # 43% vs 17%
                    'disease_driver': True,
                    'therapeutic_relevance': 'high'
                }
            }
        }
        
        # Save features
        import json
        with open(self.results_dir / "graph_enhancement_features.json", 'w') as f:
            json.dump(graph_features, f, indent=2)
        
        logger.info("Graph features saved to graph_enhancement_features.json")
        return graph_features
    
    def run_full_analysis(self):
        """Run complete analysis of iMCD experimental data"""
        logger.info("🔬 Starting iMCD experimental data analysis...")
        
        # Analyze each data type
        pathway_scores = self.analyze_proteomics_data()
        cell_features = self.analyze_scrna_data()
        functional_features = self.analyze_functional_data()
        baseline_results = self.analyze_kgml_results()
        
        # Create graph enhancement features
        graph_features = self.create_graph_features()
        
        # Summary report
        print("\n" + "="*60)
        print("📋 iMCD EXPERIMENTAL DATA SUMMARY")
        print("="*60)
        
        print("\n🎯 KEY TARGETS FOR GRAPH ENHANCEMENT:")
        print("• TNF pathway: STRONG experimental evidence")
        print("• Naive CD4+ T cells: 31x higher TNF expression")
        print("• Functional validation: 2.5x more TNF-producing cells")
        print("• Current adalimumab rank: #3 → Target: #1")
        
        print("\n📊 FEATURE ENGINEERING READY:")
        print("• Disease features: iMCD-TAFRO severity and pathways")
        print("• Drug features: TNF inhibitor mechanism matching")
        print("• Gene features: TNF expression fold changes")
        print("• Cell features: CD4+ T cell dysfunction scores")
        
        logger.info("✅ iMCD experimental data analysis complete!")
        return graph_features

def main():
    """Main function"""
    print("🔬 iMCD-TAFRO Experimental Data Analysis")
    print("=" * 50)
    
    analyzer = iMCDExperimentalDataAnalyzer()
    graph_features = analyzer.run_full_analysis()
    
    return 0

if __name__ == "__main__":
    exit(main())