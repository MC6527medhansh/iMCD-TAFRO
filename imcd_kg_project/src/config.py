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

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 EXPERIMENTAL_DATA_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

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

# Model hyperparameters (we'll tune these later)
MODEL_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 3,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 100
}

print(f"✅ Config loaded. Project root: {PROJECT_ROOT}")