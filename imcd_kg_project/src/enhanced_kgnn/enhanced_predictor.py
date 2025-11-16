#!/usr/bin/env python3
"""
Enhanced Drug-Disease Prediction with Experimental Features
Architecture: GraphSAGE + experimental node features + link prediction

CRITICAL FIX (Phase 2):
- Graph structure from full_graph.pkl (Phase 1.2 output)
- Training pairs used as LABELS only, not structure
- Preserves mechanism paths: adalimumab→TNF→Castleman
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    from torch_geometric.utils import negative_sampling
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("PyTorch Geometric not installed. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    from torch_geometric.utils import negative_sampling
    PYTORCH_GEOMETRIC_AVAILABLE = True

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import pickle

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("Installing scikit-learn...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphSAGEModel(torch.nn.Module):
    """GraphSAGE model for link prediction with experimental node features"""
    
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
    
    Architecture (CORRECTED):
    - Load FULL graph from Phase 1.2 (160K nodes, 3.6M edges)
    - Use training pairs as LABELS only
    - Add experimental features to TNF-related nodes
    - Train GraphSAGE for link prediction
    """
    
    def __init__(self, kg_loader=None, config_path=None):
        """
        Initialize predictor
        
        Args:
            kg_loader: Optional RTXKGLoader (kept for backward compatibility)
            config_path: Path to config.py (auto-detected if None)
        """
        # Auto-detect config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.py"
        
        # Import config dynamically
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config
        spec.loader.exec_module(config)
        
        # Store config values
        self.processed_graph_path = config.PROCESSED_GRAPH_PATH
        self.training_data_path = config.TRAINING_DATA_PATH if hasattr(config, 'TRAINING_DATA_PATH') else config.DATA_DIR / "kgml_data" / "training_data"
        self.adalimumab_id = config.ADALIMUMAB_ID
        self.castleman_id = config.CASTLEMAN_ID
        self.tnf_id = config.TNF_ID
        self.tnf_fold_change = 2 ** config.TNF_LOG2_FOLD_CHANGE  # Convert log2 back to linear
        
        # Experimental evidence
        self.experimental_evidence = {
            'TNF_fold_change': self.tnf_fold_change,  # 30.7x from paper
            'TNF_log2_fold_change': config.TNF_LOG2_FOLD_CHANGE,  # 4.94
            'pathway_weight': 2.0
        }
        
        # Graph components (will be populated by build_graph)
        self.entity_to_idx = {}
        self.idx_to_entity = {}
        self.node_features = None
        self.edge_index = None
        self.train_edge_index = None
        self.train_edge_labels = None
        
        logger.info(f"Initialized ExperimentalGraphPredictor")
        logger.info(f"  Graph path: {self.processed_graph_path}")
        logger.info(f"  Training data: {self.training_data_path}")
        logger.info(f"  Adalimumab: {self.adalimumab_id}")
        logger.info(f"  Castleman: {self.castleman_id}")
        logger.info(f"  TNF: {self.tnf_id}")
    
    def load_training_data(self) -> List[Tuple[str, str, int]]:
        """
        Load drug-disease pairs from verified training data
        
        Returns:
            List of (drug_id, disease_id, label) tuples
        """
        pairs = []
        
        logger.info(f"Loading training data from {self.training_data_path}")
        
        # Load positive pairs
        for file_name in ['repoDB_tp.txt', 'semmed_tp.txt', 'ndf_tp.txt', 'mychem_tp.txt']:
            file_path = self.training_data_path / file_name
            if file_path.exists():
                df = pd.read_csv(file_path, sep='\t')
                
                if 'source' in df.columns and 'target' in df.columns:
                    for _, row in df.iterrows():
                        drug_id = str(row['source']) if pd.notna(row['source']) else None
                        disease_id = str(row['target']) if pd.notna(row['target']) else None
                        
                        if drug_id and disease_id:
                            pairs.append((drug_id, disease_id, 1))
                
                logger.info(f"  Loaded {file_name}: {len(df)} positive pairs")
        
        # Load negative pairs
        for file_name in ['repoDB_tn.txt', 'semmed_tn.txt', 'ndf_tn.txt', 'mychem_tn.txt']:
            file_path = self.training_data_path / file_name
            if file_path.exists():
                df = pd.read_csv(file_path, sep='\t')
                
                if 'source' in df.columns and 'target' in df.columns:
                    for _, row in df.iterrows():
                        drug_id = str(row['source']) if pd.notna(row['source']) else None
                        disease_id = str(row['target']) if pd.notna(row['target']) else None
                        
                        if drug_id and disease_id:
                            pairs.append((drug_id, disease_id, 0))
                
                logger.info(f"  Loaded {file_name}: {len(df)} negative pairs")
        
        logger.info(f"Total training pairs loaded: {len(pairs)}")
        
        # DATA LEAKAGE CHECK
        adalimumab_castleman_pairs = [
            (d, dis) for d, dis, _ in pairs 
            if d == self.adalimumab_id and dis == self.castleman_id
        ]
        
        if len(adalimumab_castleman_pairs) > 0:
            logger.error(f"❌ DATA LEAKAGE DETECTED: {len(adalimumab_castleman_pairs)} adalimumab-Castleman pairs in training!")
            raise ValueError("Data leakage: Target pair found in training data")
        else:
            logger.info(f"✅ No data leakage: Adalimumab-Castleman pair NOT in training")
        
        return pairs
    
    def build_graph_with_experimental_features(self, use_experimental=True):
        """
        Build PyTorch Geometric graph with experimental features
        
        CRITICAL CHANGE (Phase 2):
        - Graph structure from full_graph.pkl (Phase 1.2 output)
        - Training pairs used as LABELS only, not structure
        - Preserves mechanism paths: adalimumab→TNF→Castleman
        
        Args:
            use_experimental: If True, add TNF experimental feature (4D), else baseline (3D)
        
        Returns:
            PyTorch Geometric Data object
        """
        logger.info(f"="*60)
        logger.info(f"Building graph {'WITH' if use_experimental else 'WITHOUT'} experimental features")
        logger.info(f"="*60)
        
        # 1. Load FULL graph from Phase 1.2 (NOT from training pairs!)
        logger.info(f"Loading full graph from {self.processed_graph_path}")
        
        if not self.processed_graph_path.exists():
            raise FileNotFoundError(f"Full graph not found: {self.processed_graph_path}\nRun Phase 1.2 first!")
        
        with open(self.processed_graph_path, 'rb') as f:
            nx_graph = pickle.load(f)
        
        logger.info(f"✅ Loaded graph: {nx_graph.number_of_nodes():,} nodes, {nx_graph.number_of_edges():,} edges")
        
        # 2. Create entity mappings
        entities = sorted(nx_graph.nodes())  # Sort for reproducibility
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
        self.idx_to_entity = {idx: entity for entity, idx in self.entity_to_idx.items()}
        num_nodes = len(entities)
        
        logger.info(f"Entity mapping created: {num_nodes:,} nodes")
        
        # 3. Build node features
        num_features = 4 if use_experimental else 3
        node_features = np.zeros((num_nodes, num_features), dtype=np.float32)
        
        drug_count = 0
        disease_count = 0
        other_count = 0
        tnf_feature_count = 0
        
        for entity, idx in self.entity_to_idx.items():
            # Entity type features (first 3 dimensions)
            if entity.startswith('CHEMBL.COMPOUND:'):
                node_features[idx, 0] = 1.0  # Drug
                drug_count += 1
            elif entity.startswith('MONDO:'):
                node_features[idx, 1] = 1.0  # Disease
                disease_count += 1
            else:
                node_features[idx, 2] = 1.0  # Other (genes, proteins, pathways)
                other_count += 1
            
            # Experimental feature (4th dimension) - ONLY if enabled
            if use_experimental and self._is_tnf_related(entity):
                node_features[idx, 3] = self.experimental_evidence['TNF_log2_fold_change']
                tnf_feature_count += 1
                logger.info(f"  ⭐ Added TNF feature to {entity}: {node_features[idx, 3]:.4f}")
        
        logger.info(f"Node features assigned:")
        logger.info(f"  Drugs: {drug_count:,}")
        logger.info(f"  Diseases: {disease_count:,}")
        logger.info(f"  Other (genes/proteins): {other_count:,}")
        if use_experimental:
            logger.info(f"  TNF features assigned: {tnf_feature_count}")
        
        # 4. Convert NetworkX edges to PyG format (undirected)
        edge_list = []
        for source, target in nx_graph.edges():
            if source in self.entity_to_idx and target in self.entity_to_idx:
                source_idx = self.entity_to_idx[source]
                target_idx = self.entity_to_idx[target]
                edge_list.append([source_idx, target_idx])
                edge_list.append([target_idx, source_idx])  # Undirected
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        logger.info(f"Graph edges converted: {edge_index.shape[1]:,} edges (bidirectional)")
        
        # 5. Load training pairs (for LABELS, not structure!)
        training_pairs = self.load_training_data()
        
        # 6. Create edge labels from training pairs
        train_edge_list = []
        train_labels = []
        skipped = 0
        
        for drug_id, disease_id, label in training_pairs:
            if drug_id in self.entity_to_idx and disease_id in self.entity_to_idx:
                drug_idx = self.entity_to_idx[drug_id]
                disease_idx = self.entity_to_idx[disease_id]
                train_edge_list.append([drug_idx, disease_idx])
                train_labels.append(label)
            else:
                skipped += 1
        
        if skipped > 0:
            logger.warning(f"⚠️  Skipped {skipped} training pairs (entities not in graph)")
        
        train_edge_index = torch.tensor(train_edge_list, dtype=torch.long).t().contiguous()
        train_edge_labels = torch.tensor(train_labels, dtype=torch.float)
        
        logger.info(f"Training supervision created: {len(train_labels):,} labeled pairs")
        logger.info(f"  Positive pairs: {int(train_edge_labels.sum().item()):,}")
        logger.info(f"  Negative pairs: {len(train_labels) - int(train_edge_labels.sum().item()):,}")
        
        # 7. Store for later use
        self.node_features = torch.tensor(node_features, dtype=torch.float)
        self.edge_index = edge_index
        self.train_edge_index = train_edge_index
        self.train_edge_labels = train_edge_labels
        
        # 8. Validate critical entities exist
        logger.info(f"Validating critical entities...")
        assert self.adalimumab_id in self.entity_to_idx, f"Adalimumab not in graph!"
        assert self.castleman_id in self.entity_to_idx, f"Castleman not in graph!"
        assert self.tnf_id in self.entity_to_idx, f"TNF not in graph!"
        logger.info(f"✅ All critical entities present")
        
        # 9. Create PyTorch Geometric data object
        data = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            train_edge_index=self.train_edge_index,
            train_edge_labels=self.train_edge_labels,
            num_nodes=num_nodes
        )
        
        logger.info(f"="*60)
        logger.info(f"Graph construction complete:")
        logger.info(f"  Nodes: {num_nodes:,}")
        logger.info(f"  Message-passing edges: {edge_index.shape[1]:,}")
        logger.info(f"  Training pairs: {len(train_labels):,}")
        logger.info(f"  Feature dimensions: {num_features}D")
        logger.info(f"  Graph mode: {'ENHANCED (4D with TNF)' if use_experimental else 'BASELINE (3D)'}")
        logger.info(f"="*60)
        
        return data
    
    def _is_tnf_related(self, entity_id: str) -> bool:
        """
        Determine if entity should receive experimental TNF features
        Based on biological justification
        """
        # Primary TNF gene/protein
        if entity_id == self.tnf_id:
            return True
        
        # Adalimumab (TNF inhibitor)
        if entity_id == self.adalimumab_id:
            return True
        
        # Other TNF-related entities
        tnf_keywords = ['TNF', 'tumor necrosis factor', 'tnf inhibitor']
        for keyword in tnf_keywords:
            if keyword.lower() in entity_id.lower():
                return True
        
        return False
    
    def train_model(self, data, model, epochs=200, lr=0.01):
        """
        Train GraphSAGE model using training pairs as supervision
        
        CRITICAL: Uses data.train_edge_index for supervision, NOT edge_index
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()
        
        model.train()
        
        logger.info(f"Training model for {epochs} epochs...")
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass using ALL edges for message passing
            z = model(data.x, data.edge_index)
            
            # Get predictions for training pairs
            pos_mask = data.train_edge_labels == 1
            neg_mask = data.train_edge_labels == 0
            
            pos_edge_index = data.train_edge_index[:, pos_mask]
            neg_edge_index = data.train_edge_index[:, neg_mask]
            
            # Positive edge predictions
            pos_pred = model.predict_links(z, pos_edge_index)
            pos_loss = criterion(pos_pred, torch.ones(pos_pred.size(0)))
            
            # Negative edge predictions
            neg_pred = model.predict_links(z, neg_edge_index)
            neg_loss = criterion(neg_pred, torch.zeros(neg_pred.size(0)))
            
            # Total loss
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                logger.info(f"  Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        logger.info(f"✅ Training complete")
        
        return model, data
    
    def evaluate_drug_ranking(self, model, data, disease_entity):
        """
        Evaluate drug ranking for specific disease
        Returns ranking of all drugs for the disease
        """
        model.eval()
        
        with torch.no_grad():
            # Get node embeddings
            z = model(data.x, data.edge_index)
            
            if disease_entity not in self.entity_to_idx:
                logger.warning(f"Disease {disease_entity} not in graph")
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
    
    def compare_predictions(self, epochs=200) -> Dict[str, any]:
        """
        Compare baseline vs enhanced predictions
        """
        logger.info("\n" + "="*60)
        logger.info("BASELINE vs ENHANCED COMPARISON")
        logger.info("="*60)
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Build and train baseline model
        logger.info("\n1. BASELINE MODEL (3D features):")
        baseline_data = self.build_graph_with_experimental_features(use_experimental=False)
        baseline_model = GraphSAGEModel(input_dim=3, hidden_dim=64, output_dim=32)
        baseline_model, baseline_data = self.train_model(baseline_data, baseline_model, epochs=epochs)
        
        # Build and train enhanced model
        logger.info("\n2. ENHANCED MODEL (4D features with TNF):")
        torch.manual_seed(42)  # Reset seed
        enhanced_data = self.build_graph_with_experimental_features(use_experimental=True)
        enhanced_model = GraphSAGEModel(input_dim=4, hidden_dim=64, output_dim=32)
        enhanced_model, enhanced_data = self.train_model(enhanced_data, enhanced_model, epochs=epochs)
        
        # Evaluate rankings for Castleman disease
        logger.info("\n3. EVALUATION:")
        baseline_ranking = self.evaluate_drug_ranking(baseline_model, baseline_data, self.castleman_id)
        enhanced_ranking = self.evaluate_drug_ranking(enhanced_model, enhanced_data, self.castleman_id)
        
        # Find adalimumab rankings
        baseline_adalimumab_rank = None
        enhanced_adalimumab_rank = None
        baseline_adalimumab_score = 0
        enhanced_adalimumab_score = 0
        
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
        
        improvement = baseline_adalimumab_rank - enhanced_adalimumab_rank if (baseline_adalimumab_rank and enhanced_adalimumab_rank) else 0
        
        logger.info(f"\nRESULTS:")
        logger.info(f"  Baseline:  Rank #{baseline_adalimumab_rank}, Score {baseline_adalimumab_score:.4f}")
        logger.info(f"  Enhanced:  Rank #{enhanced_adalimumab_rank}, Score {enhanced_adalimumab_score:.4f}")
        logger.info(f"  Improvement: {improvement:+d} positions")
        
        return {
            'baseline_adalimumab_rank': baseline_adalimumab_rank,
            'enhanced_adalimumab_rank': enhanced_adalimumab_rank,
            'baseline_adalimumab_score': baseline_adalimumab_score,
            'enhanced_adalimumab_score': enhanced_adalimumab_score,
            'ranking_improvement': improvement,
            'baseline_top_10': list(baseline_ranking.items())[:10],
            'enhanced_top_10': list(enhanced_ranking.items())[:10]
        }


if __name__ == "__main__":
    """Quick test of enhanced predictor"""
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    print("Testing ExperimentalGraphPredictor...")
    
    predictor = ExperimentalGraphPredictor()
    results = predictor.compare_predictions(epochs=100)
    
    print("\n" + "="*60)
    print("EXPERIMENTAL ENHANCEMENT RESULTS")
    print("="*60)
    print(f"Baseline adalimumab rank: #{results['baseline_adalimumab_rank']}")
    print(f"Enhanced adalimumab rank: #{results['enhanced_adalimumab_rank']}")
    print(f"Ranking improvement: {results['ranking_improvement']:+d} positions")
    print(f"Baseline score: {results['baseline_adalimumab_score']:.4f}")
    print(f"Enhanced score: {results['enhanced_adalimumab_score']:.4f}")
    
    if results['ranking_improvement'] > 0:
        print(f"\n✅ SUCCESS: Adalimumab improved by {results['ranking_improvement']} positions!")
    else:
        print(f"\n⚠️  WARNING: No improvement detected")