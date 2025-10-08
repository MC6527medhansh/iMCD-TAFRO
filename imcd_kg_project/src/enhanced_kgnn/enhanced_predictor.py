#!/usr/bin/env python3
"""
Enhanced Drug-Disease Prediction with Experimental Features
Architecture: GraphSAGE + experimental node features + link prediction
"""

import os
# Fix OpenMP issue before importing torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if PyTorch Geometric is installed
try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    from torch_geometric.utils import train_test_split_edges
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    from torch_geometric.utils import train_test_split_edges
    PYTORCH_GEOMETRIC_AVAILABLE = True
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Check for scikit-learn
try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.metrics import roc_auc_score

from rtx_kg_loader import RTXKGLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphSAGEModel(torch.nn.Module):
    """
    GraphSAGE model for link prediction with experimental node features
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def predict_links(self, z, edge_index):
        """Predict link probabilities between node pairs"""
        row, col = edge_index
        return torch.sigmoid(torch.sum(z[row] * z[col], dim=1))

class ExperimentalGraphPredictor:
    """
    Enhanced drug-disease prediction using experimental evidence
    
    Architecture:
    - Build graph from verified training data
    - Add experimental features to TNF-related nodes
    - Train GraphSAGE for link prediction
    - Compare baseline vs enhanced rankings
    """
    
    def __init__(self, kg_loader: RTXKGLoader):
        self.kg_loader = kg_loader
        
        # Experimental evidence from iMCD paper
        self.experimental_evidence = {
            'TNF_fold_change': 30.7,  # CD4+ T cells
            'functional_validation': 2.53,  # Functional assay result
            'pathway_weight': 2.0
        }
        
        # Verified entities from our analysis
        self.adalimumab_id = "CHEMBL.COMPOUND:CHEMBL1201580"
        self.castleman_id = "MONDO:0015564"
        self.tnf_inhibitor_class = "UMLS:C3653350"
        
        # Training data paths
        self.training_data_path = Path("../../data/kgml_data/training_data")
        
        # Graph components
        self.entity_to_idx = {}
        self.idx_to_entity = {}
        self.node_features = None
        self.edge_index = None
        self.edge_labels = None
        
    def load_training_data(self) -> List[Tuple[str, str, int]]:
        """
        Load drug-disease pairs from verified training data
        Returns: List of (drug_id, disease_id, label) tuples
        """
        pairs = []
        
        # Load positive pairs from RepoDB and SemMed
        for file_name in ['repoDB_tp.txt', 'semmed_tp.txt']:
            file_path = self.training_data_path / file_name
            if file_path.exists():
                df = pd.read_csv(file_path, sep='\t')
                for _, row in df.iterrows():
                    if len(row) >= 2:
                        drug_id = row.iloc[0] if pd.notna(row.iloc[0]) else None
                        disease_id = row.iloc[1] if pd.notna(row.iloc[1]) else None
                        # Ensure both are strings and not empty
                        if drug_id and disease_id and isinstance(drug_id, str) and isinstance(disease_id, str):
                            pairs.append((drug_id, disease_id, 1))
        
        # Load negative pairs
        for file_name in ['repoDB_tn.txt', 'semmed_tn.txt']:
            file_path = self.training_data_path / file_name
            if file_path.exists():
                df = pd.read_csv(file_path, sep='\t')
                for _, row in df.iterrows():
                    if len(row) >= 2:
                        drug_id = row.iloc[0] if pd.notna(row.iloc[0]) else None
                        disease_id = row.iloc[1] if pd.notna(row.iloc[1]) else None
                        # Ensure both are strings and not empty
                        if drug_id and disease_id and isinstance(drug_id, str) and isinstance(disease_id, str):
                            pairs.append((drug_id, disease_id, 0))
        
        logger.info(f"Loaded {len(pairs)} drug-disease pairs")
        return pairs
    
    def build_graph_with_experimental_features(self, use_experimental=True):
        """
        Build PyTorch Geometric graph with experimental node features
        """
        logger.info("Building graph with experimental features...")
        
        # Load training pairs
        training_pairs = self.load_training_data()
        
        # Extract unique entities
        entities = set()
        edges = []
        labels = []
        
        for drug_id, disease_id, label in training_pairs:
            entities.add(drug_id)
            entities.add(disease_id)
            edges.append((drug_id, disease_id))
            labels.append(label)
        
        # Create entity mappings
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(entities))}
        self.idx_to_entity = {idx: entity for entity, idx in self.entity_to_idx.items()}
        
        num_nodes = len(entities)
        
        # Create base node features (one-hot encoding + entity type)
        node_features = np.zeros((num_nodes, 4))  # Base + experimental features
        
        for entity, idx in self.entity_to_idx.items():
            # Entity type features
            if entity.startswith('CHEMBL.COMPOUND:'):
                node_features[idx, 0] = 1.0  # Drug
            elif entity.startswith('MONDO:'):
                node_features[idx, 1] = 1.0  # Disease
            else:
                node_features[idx, 2] = 1.0  # Other
            
            # Add experimental features if enabled
            if use_experimental:
                # TNF-related entities get experimental features
                if self._is_tnf_related(entity):
                    node_features[idx, 3] = np.log2(self.experimental_evidence['TNF_fold_change'])
                    logger.info(f"Added experimental feature to {entity}")
        
        # Convert edges to tensor format
        edge_pairs = []
        edge_labels = []
        
        for (drug_id, disease_id), label in zip(edges, labels):
            drug_idx = self.entity_to_idx[drug_id]
            disease_idx = self.entity_to_idx[disease_id]
            edge_pairs.append([drug_idx, disease_idx])
            edge_pairs.append([disease_idx, drug_idx])  # Undirected
            edge_labels.extend([label, label])
        
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        self.node_features = torch.tensor(node_features, dtype=torch.float)
        self.edge_index = edge_index
        self.edge_labels = torch.tensor(edge_labels, dtype=torch.float)
        
        # Create PyTorch Geometric data object
        data = Data(x=self.node_features, edge_index=self.edge_index, y=self.edge_labels)
        return data
    
    def _is_tnf_related(self, entity_id: str) -> bool:
        """
        Determine if entity should receive experimental TNF features
        Based on biological justification
        """
        tnf_keywords = ['TNF', 'tumor necrosis factor', 'adalimumab', 'tnf inhibitor']
        
        # Check our verified TNF-related entities
        if entity_id == self.adalimumab_id:
            return True
        if entity_id == self.tnf_inhibitor_class:
            return True
        
        # Check for TNF-related terms
        for keyword in tnf_keywords:
            if keyword.lower() in entity_id.lower():
                return True
        
        return False
    
    def train_model(self, data, model, epochs=200):
        """Train GraphSAGE model"""
        data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            z = model(data.x, data.train_pos_edge_index)
            
            # Positive edges
            pos_pred = model.predict_links(z, data.train_pos_edge_index)
            pos_loss = criterion(pos_pred, torch.ones(pos_pred.size(0)))
            
            # Negative edges
            neg_pred = model.predict_links(z, data.train_neg_edge_index)
            neg_loss = criterion(neg_pred, torch.zeros(neg_pred.size(0)))
            
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return model, data
    
    def evaluate_drug_ranking(self, model, data, disease_entity):
        """
        Evaluate drug ranking for specific disease
        Returns ranking of all drugs for the disease
        """
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.train_pos_edge_index)
            
            if disease_entity not in self.entity_to_idx:
                logger.warning(f"Disease {disease_entity} not in training data")
                return {}
            
            disease_idx = self.entity_to_idx[disease_entity]
            drug_scores = {}
            
            # Score all drugs for this disease
            for entity, idx in self.entity_to_idx.items():
                if entity.startswith('CHEMBL.COMPOUND:'):
                    edge = torch.tensor([[idx], [disease_idx]], dtype=torch.long)
                    score = model.predict_links(z, edge).item()
                    drug_scores[entity] = score
            
            # Sort by score
            ranked_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
            return dict(ranked_drugs)
    
    def compare_predictions(self) -> Dict[str, any]:
        """
        Compare baseline vs enhanced predictions
        """
        logger.info("Training baseline model (no experimental features)...")
        baseline_data = self.build_graph_with_experimental_features(use_experimental=False)
        baseline_model = GraphSAGEModel(input_dim=3, hidden_dim=64, output_dim=32)
        baseline_model, baseline_data = self.train_model(baseline_data, baseline_model)
        
        logger.info("Training enhanced model (with experimental features)...")
        enhanced_data = self.build_graph_with_experimental_features(use_experimental=True)
        enhanced_model = GraphSAGEModel(input_dim=4, hidden_dim=64, output_dim=32)
        enhanced_model, enhanced_data = self.train_model(enhanced_data, enhanced_model)
        
        # Evaluate rankings for Castleman disease
        baseline_ranking = self.evaluate_drug_ranking(baseline_model, baseline_data, self.castleman_id)
        enhanced_ranking = self.evaluate_drug_ranking(enhanced_model, enhanced_data, self.castleman_id)
        
        # Find adalimumab rankings
        baseline_adalimumab_rank = None
        enhanced_adalimumab_rank = None
        
        for rank, (drug, score) in enumerate(baseline_ranking.items(), 1):
            if drug == self.adalimumab_id:
                baseline_adalimumab_rank = rank
                baseline_adalimumab_score = score
                break
        
        for rank, (drug, score) in enumerate(enhanced_ranking.items(), 1):
            if drug == self.adalimumab_id:
                enhanced_adalimumab_rank = rank
                enhanced_adalimumab_score = score
                break
        
        return {
            'baseline_adalimumab_rank': baseline_adalimumab_rank,
            'enhanced_adalimumab_rank': enhanced_adalimumab_rank,
            'baseline_adalimumab_score': baseline_adalimumab_score if baseline_adalimumab_rank else 0,
            'enhanced_adalimumab_score': enhanced_adalimumab_score if enhanced_adalimumab_rank else 0,
            'ranking_improvement': (baseline_adalimumab_rank - enhanced_adalimumab_rank) if baseline_adalimumab_rank and enhanced_adalimumab_rank else 0,
            'baseline_top_10': list(baseline_ranking.items())[:10],
            'enhanced_top_10': list(enhanced_ranking.items())[:10]
        }

if __name__ == "__main__":
    # Test the enhanced predictor
    kg_path = Path("../../data/kgml_data/bkg_rtxkg2c_v2.7.3")
    loader = RTXKGLoader(kg_path)
    
    predictor = ExperimentalGraphPredictor(loader)
    results = predictor.compare_predictions()
    
    print("=== EXPERIMENTAL ENHANCEMENT RESULTS ===")
    print(f"Baseline adalimumab rank: {results['baseline_adalimumab_rank']}")
    print(f"Enhanced adalimumab rank: {results['enhanced_adalimumab_rank']}")
    print(f"Ranking improvement: {results['ranking_improvement']} positions")
    print(f"Baseline score: {results['baseline_adalimumab_score']:.4f}")
    print(f"Enhanced score: {results['enhanced_adalimumab_score']:.4f}")
    
    print("\nBaseline top 10:")
    for i, (drug, score) in enumerate(results['baseline_top_10'], 1):
        print(f"{i}. {drug}: {score:.4f}")
    
    print("\nEnhanced top 10:")
    for i, (drug, score) in enumerate(results['enhanced_top_10'], 1):
        print(f"{i}. {drug}: {score:.4f}")
    
    if results['ranking_improvement'] > 0:
        print(f"\nSUCCESS: Adalimumab improved by {results['ranking_improvement']} positions!")
    else:
        print(f"\nWARNING: No improvement detected")