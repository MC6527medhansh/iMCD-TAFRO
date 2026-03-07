"""
Configuration file for iMCD-KG project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src" 
RESULTS_DIR = PROJECT_ROOT / "results"
LITERATURE_DIR = PROJECT_ROOT / "literature"

# External repositories
KGML_REPO = SRC_DIR / "KGML-xDTD"
TXGNN_REPO = SRC_DIR / "TxGNN"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXPERIMENTAL_DATA_DIR = DATA_DIR / "experimental"

# Phase 1.2 output (full graph from RTX-KG2)
PROCESSED_GRAPH_PATH = PROCESSED_DATA_DIR / "full_graph.pkl"

# Training data (drug-disease pairs for supervision)
TRAINING_DATA_PATH = DATA_DIR / "kgml_data" / "training_data"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 EXPERIMENTAL_DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Critical entity IDs (verified in Phase 1.1)
ADALIMUMAB_ID = "CHEMBL.COMPOUND:CHEMBL1201580"
CASTLEMAN_ID = "MONDO:0015564"
TNF_ID = "UniProtKB:P01375"

# Experimental data from iMCD-TAFRO paper
TNF_LOG2_FOLD_CHANGE = 4.94  # log2(30.7x upregulation in naive CD4+ T cells)
TNF_LINEAR_FOLD_CHANGE = 30.7  # Linear fold change

# Test disease sets (for generalization testing)
TNF_DISEASES = {
    'MONDO:0015564': 'Castleman Disease (iMCD-TAFRO)',
    'MONDO:0008383': 'Rheumatoid Arthritis',
    'MONDO:0005011': 'Crohn Disease',
    'MONDO:0005083': 'Psoriasis'
}

NON_TNF_DISEASES = {
    'MONDO:0005148': 'Type 2 Diabetes',
    'MONDO:0005044': 'Hypertension',
    'MONDO:0011382': 'Alzheimer Disease'
}

# iMCD-TAFRO experimental data from the paper
IMCD_EXPERIMENTAL_DATA = {
    'proteomics': {
        'method': 'SOMAscan v4.1',
        'samples': {'imcd': 26, 'healthy': 15},
        'analytes': 6408,
        'key_pathways': ['TNF-via-NF-κB', 'IL-6-JAK-STAT3', 'inflammatory_response']
    },
    'scrna_seq': {
        'naive_cd4_tnf_fold_change': 4.94,  # log2
        'samples': {'flare': 3, 'healthy': 2},
        'key_finding': 'TNF expression 31x higher in naive CD4+ T cells'
    },
    'functional_assays': {
        'naive_cd4_tnf_producers': {
            'imcd_percent': 43,
            'healthy_percent': 17, 
            'p_value': 0.01
        }
    },
    'kgml_results': {
        'adalimumab_score': 0.83736,
        'adalimumab_rank': 3,
        'top_drugs': ['tocilizumab', 'siltuximab', 'adalimumab']
    }
}

# Model hyperparameters
MODEL_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 3,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 100
}

print(f"✅ Config loaded. Project root: {PROJECT_ROOT}")
