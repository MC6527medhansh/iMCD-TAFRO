"""
Literature summary and key insights for iMCD-KG project
"""

import sys
sys.path.append('..')
from config import IMCD_EXPERIMENTAL_DATA

def print_disease_paper_summary():
    """Summary of iMCD-TAFRO disease paper (NEJM)"""
    print("="*60)
    print("📄 iMCD-TAFRO Disease Paper (NEJM + Supplement)")
    print("="*60)
    
    print("\n🎯 KEY FINDING:")
    print("TNF signaling from CD4+ T cells is a targetable pathway")
    print("Adalimumab successfully treated refractory patient for 24+ months")
    
    print("\n🔬 EXPERIMENTAL EVIDENCE:")
    data = IMCD_EXPERIMENTAL_DATA
    print(f"• Proteomics: {data['proteomics']['samples']['imcd']} patients vs {data['proteomics']['samples']['healthy']} controls")
    print(f"• scRNA-seq: TNF {data['scrna_seq']['naive_cd4_tnf_fold_change']} log2 fold higher in naive CD4+ T cells")
    print(f"• Flow cytometry: {data['functional_assays']['naive_cd4_tnf_producers']['imcd_percent']}% vs {data['functional_assays']['naive_cd4_tnf_producers']['healthy_percent']}% TNF+ cells")
    
    print(f"\n🤖 AI PREDICTION:")
    print(f"KGML-xDTD ranked adalimumab #{data['kgml_results']['adalimumab_rank']} (score: {data['kgml_results']['adalimumab_score']})")

def print_kg_methods_summary():
    """Summary of knowledge graph methods"""
    print("\n" + "="*60)
    print("🕸️  Knowledge Graph Methods")
    print("="*60)
    
    print("\n📊 KGML-xDTD (GigaScience 2023):")
    print("• Purpose: Drug repurposing + mechanism explanation")
    print("• Method: GraphSAGE + Reinforcement Learning for MOA paths")
    print("• Strength: Provides testable biological pathways")
    print("• Data: RTX-KG2 (3.7M nodes, 18.3M edges)")
    
    print("\n🔮 TxGNN (Nature Medicine 2024):")
    print("• Purpose: Zero-shot drug repurposing")  
    print("• Method: GNN + metric learning for disease similarity")
    print("• Strength: Works for diseases with no known treatments")
    print("• Data: 17,080 diseases, 7,957 drugs")

def print_project_goals():
    """Print our project objectives"""
    print("\n" + "="*60)
    print("🎯 OUR PROJECT GOALS")
    print("="*60)
    
    print("\n🔥 PRIMARY OBJECTIVE:")
    print("Integrate experimental gene expression data with knowledge graphs")
    print("to improve drug ranking for rare diseases like iMCD-TAFRO")
    
    print("\n📈 SUCCESS METRIC:")
    print("Make adalimumab rank HIGHER than position #3 for iMCD-TAFRO")
    print("(currently ranks #3 in KGML-xDTD)")
    
    print("\n🧠 APPROACH IDEAS:")
    print("• Add experimental features as node attributes")
    print("• Use TNF pathway evidence to weight drug-disease edges")
    print("• Graph transformer to attend to experimental evidence")
    print("• Message passing with experimental data integration")

def main():
    """Print complete literature summary"""
    print("📚 iMCD-KG Project Literature Summary")
    print("=" * 80)
    
    print_disease_paper_summary()
    print_kg_methods_summary() 
    print_project_goals()
    
    print("\n" + "="*80)
    print("✅ Literature review complete. Ready for implementation!")
    print("📅 Next: Day 2 - Single-cell RNA-seq deep dive")

if __name__ == "__main__":
    main()