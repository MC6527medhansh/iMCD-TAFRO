#!/usr/bin/env python3
"""
gat_predictor.py
----------------
Purpose: GATConv-based drug-disease prediction with edge weights on
         disease-protein connections. Replaces the failed GraphSAGE
         node-feature injection approach.

Why GraphSAGE node features failed (confirmed by diagnostics Jan 2026):
  - Enhanced rank ranged #4,780 to #65,777 across 5 seeds (unstable)
  - Non-TNF diseases improved MORE than TNF diseases (not disease-specific)
  - Root cause: adalimumab-Castleman cosine similarity = 0.076 even after
    enhancement. Scoring is drug-disease dot product, so injecting a feature
    on adalimumab/TNF nodes does nothing useful for Castleman ranking.

Why GATConv + edge weights works:
  - TNF->Castleman is a direct edge (1 hop, confirmed in graph)
  - Adalimumab->TNF is a direct edge (1 hop, confirmed in graph)
  - GATConv attention: alpha_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j || W_e e_ij]))
  - High weight on TNF->Castleman edge -> Castleman embedding dominated by TNF
  - TNF embedding is shaped by adalimumab (its direct neighbor)
  - After 2 GATConv layers: adalimumab-Castleman similarity rises via TNF bridge
  - For diabetes/hypertension: TNF->disease weight = 1.0 -> no preferential signal
  - Disease-specific by construction

Validation strategy:
  Step 1 - Random weights (this file supports it): confirm GATConv reads edge_attr
  Step 2 - TNF fold-change (30.7x): confirm adalimumab ranks higher for TNF diseases
  Step 3 - Real scRNA-seq data (pending from Prof. Singh): cell-type-stratified weights

Graph facts (confirmed from Sockeye diagnostics Jan 16 2026):
  - 160,936 nodes, 3,626,585 edges (7,253,170 bidirectional)
  - PyG version 2.7.0 (GATConv with edge_dim is available)
  - PyTorch 2.9.0+cpu (no CUDA in current conda env)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import pickle
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GATModel(torch.nn.Module):
    """
    Two-layer GATConv for link prediction.

    Accepts edge_attr (shape [E, 1]) in forward(). The attention mechanism
    in GATConv incorporates edge features into the attention coefficient
    computation, so edge weights directly influence which neighbors'
    information is amplified when updating each node's embedding.

    Default: hidden_dim=64, output_dim=32, heads=4, edge_dim=1.
    hidden_dim must be divisible by heads.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        heads: int = 4,
        edge_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
            )
        self.conv1 = GATConv(
            input_dim,
            hidden_dim // heads,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=True,
        )
        self.conv2 = GATConv(
            hidden_dim,
            output_dim,
            heads=1,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=False,
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:          Node features [N, input_dim]
            edge_index: Edge indices   [2, E]
            edge_attr:  Edge weights   [E, 1] — must be provided for edge weights
                        to influence attention. If None, GATConv ignores edges.
        Returns:
            Node embeddings [N, output_dim]
        """
        x = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x

    def predict_links(
        self, z: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Dot-product link prediction. Returns sigmoid(h_i . h_j) for each edge.
        Output is in [0, 1].
        """
        row, col = edge_index
        return torch.sigmoid(torch.sum(z[row] * z[col], dim=1))


class GATPredictor:
    """
    End-to-end predictor: graph loading -> edge weight assignment ->
    GATConv training -> drug ranking.

    Call sequence:
        predictor = GATPredictor()
        data = predictor.build_graph()                          # loads nx_graph
        neighbors = predictor.get_disease_protein_neighbors(predictor.castleman_id)
        weights = predictor.make_random_weights(neighbors, seed=42)  # or make_tnf_weights
        data_weighted = predictor.build_graph(disease_edge_weights=weights)
        model = GATModel(input_dim=3)
        model = predictor.train_model(data_weighted, model, epochs=200)
        rank, score = predictor.get_adalimumab_rank(model, data_weighted,
                                                     predictor.castleman_id)
    """

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.py"

        import sys
        import importlib.util

        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        sys.modules["config"] = config
        spec.loader.exec_module(config)

        self.processed_graph_path = config.PROCESSED_GRAPH_PATH
        self.training_data_path = (
            config.TRAINING_DATA_PATH
            if hasattr(config, "TRAINING_DATA_PATH")
            else config.DATA_DIR / "kgml_data" / "training_data"
        )
        self.adalimumab_id = config.ADALIMUMAB_ID
        self.castleman_id = config.CASTLEMAN_ID
        self.tnf_id = config.TNF_ID
        self.tnf_fold_change = 2 ** config.TNF_LOG2_FOLD_CHANGE  # 30.7x linear

        # Populated by build_graph()
        self.entity_to_idx: Dict[str, int] = {}
        self.idx_to_entity: Dict[int, str] = {}
        self.nx_graph = None

        logger.info("Initialized GATPredictor")
        logger.info(f"  Adalimumab: {self.adalimumab_id}")
        logger.info(f"  Castleman:  {self.castleman_id}")
        logger.info(f"  TNF:        {self.tnf_id}")
        logger.info(f"  TNF fold change (linear): {self.tnf_fold_change:.1f}x")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_training_data(self) -> List[Tuple[str, str, int]]:
        """
        Load drug-disease supervision pairs from RepoDB and related sources.
        Raises ValueError if adalimumab-Castleman pair is found (data leakage).
        """
        pairs = []
        for fname, label in [
            ("repoDB_tp.txt", 1),
            ("semmed_tp.txt", 1),
            ("ndf_tp.txt", 1),
            ("mychem_tp.txt", 1),
            ("repoDB_tn.txt", 0),
            ("semmed_tn.txt", 0),
            ("ndf_tn.txt", 0),
            ("mychem_tn.txt", 0),
        ]:
            fpath = self.training_data_path / fname
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath, sep="\t")
            if "source" in df.columns and "target" in df.columns:
                for _, row in df.iterrows():
                    src = str(row["source"]) if pd.notna(row["source"]) else None
                    tgt = str(row["target"]) if pd.notna(row["target"]) else None
                    if src and tgt:
                        pairs.append((src, tgt, label))

        leakage = [
            (d, dis)
            for d, dis, _ in pairs
            if d == self.adalimumab_id and dis == self.castleman_id
        ]
        if leakage:
            raise ValueError(
                f"Data leakage detected: {len(leakage)} adalimumab-Castleman "
                "pairs found in training data."
            )

        logger.info(f"Loaded {len(pairs):,} training pairs — no data leakage")
        return pairs

    # ------------------------------------------------------------------
    # Graph building
    # ------------------------------------------------------------------

    def build_graph(
        self,
        disease_edge_weights: Optional[Dict[str, float]] = None,
        default_weight: float = 1.0,
    ) -> Data:
        """
        Build PyG Data object from full_graph.pkl with edge weight attributes.

        Args:
            disease_edge_weights:
                Dict mapping protein_id -> float weight for edges that
                connect directly to the Castleman disease node.
                All other edges receive default_weight.
                Pass None to get uniform weights (baseline — no signal).
            default_weight:
                Weight assigned to edges not in disease_edge_weights.
                Default 1.0 is neutral (no amplification).

        Returns:
            PyG Data object with fields:
              x                 [N, 3]  one-hot: drug(0) / disease(1) / other(2)
              edge_index        [2, E]  bidirectional edge list
              edge_attr         [E, 1]  edge weights (float)
              train_edge_index  [2, T]  drug-disease supervision pairs
              train_edge_labels [T]     1=positive treatment, 0=negative

        Note: entity_to_idx and idx_to_entity are populated as a side effect.
        """
        logger.info("=" * 60)
        if disease_edge_weights:
            logger.info(
                f"Building graph with {len(disease_edge_weights)} "
                "weighted disease edges"
            )
        else:
            logger.info(
                f"Building graph: uniform weight {default_weight} (baseline)"
            )
        logger.info("=" * 60)

        if not self.processed_graph_path.exists():
            raise FileNotFoundError(
                f"Graph file not found: {self.processed_graph_path}\n"
                "Phase 1.2 must be run first to generate full_graph.pkl"
            )

        with open(self.processed_graph_path, "rb") as f:
            self.nx_graph = pickle.load(f)

        logger.info(
            f"Loaded graph: {self.nx_graph.number_of_nodes():,} nodes, "
            f"{self.nx_graph.number_of_edges():,} edges"
        )

        # Sorted for reproducible index assignment
        entities = sorted(self.nx_graph.nodes())
        self.entity_to_idx = {e: i for i, e in enumerate(entities)}
        self.idx_to_entity = {i: e for e, i in self.entity_to_idx.items()}
        num_nodes = len(entities)

        # 3D one-hot node features
        node_features = np.zeros((num_nodes, 3), dtype=np.float32)
        counts = {"drug": 0, "disease": 0, "other": 0}
        for entity, idx in self.entity_to_idx.items():
            if entity.startswith("CHEMBL.COMPOUND:"):
                node_features[idx, 0] = 1.0
                counts["drug"] += 1
            elif entity.startswith("MONDO:"):
                node_features[idx, 1] = 1.0
                counts["disease"] += 1
            else:
                node_features[idx, 2] = 1.0
                counts["other"] += 1
        logger.info(
            f"Node types: {counts['drug']:,} drugs, "
            f"{counts['disease']:,} diseases, {counts['other']:,} other"
        )

        # Build edge_index and edge_attr
        edge_src: List[int] = []
        edge_dst: List[int] = []
        edge_weights: List[float] = []
        weighted_count = 0

        for source, target in self.nx_graph.edges():
            if source not in self.entity_to_idx or target not in self.entity_to_idx:
                continue

            src_idx = self.entity_to_idx[source]
            tgt_idx = self.entity_to_idx[target]
            weight = default_weight

            if disease_edge_weights:
                # Disease -> protein edge (forward direction in nx_graph)
                if (
                    source == self.castleman_id
                    and target in disease_edge_weights
                ):
                    weight = disease_edge_weights[target]
                    weighted_count += 1
                # Protein -> disease edge (reverse direction in nx_graph)
                elif (
                    target == self.castleman_id
                    and source in disease_edge_weights
                ):
                    weight = disease_edge_weights[source]
                    weighted_count += 1

            # Add both directions (undirected message passing)
            edge_src.extend([src_idx, tgt_idx])
            edge_dst.extend([tgt_idx, src_idx])
            edge_weights.extend([weight, weight])

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        # GATConv expects edge_attr shape [E, edge_dim] where edge_dim=1
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        logger.info(f"edge_index shape: {edge_index.shape}")
        logger.info(f"edge_attr shape:  {edge_attr.shape}")
        logger.info(
            f"Edge weight stats: min={edge_attr.min().item():.3f}, "
            f"max={edge_attr.max().item():.3f}, "
            f"mean={edge_attr.mean().item():.3f}"
        )
        if disease_edge_weights:
            logger.info(f"Non-default disease edges applied: {weighted_count}")

        # Training supervision (labels only, not graph structure)
        training_pairs = self._load_training_data()
        train_edges: List[List[int]] = []
        train_labels: List[int] = []
        skipped = 0

        for drug_id, disease_id, label in training_pairs:
            if drug_id in self.entity_to_idx and disease_id in self.entity_to_idx:
                train_edges.append(
                    [self.entity_to_idx[drug_id], self.entity_to_idx[disease_id]]
                )
                train_labels.append(label)
            else:
                skipped += 1

        if skipped:
            logger.warning(f"Skipped {skipped:,} training pairs (not in graph)")

        train_edge_index = (
            torch.tensor(train_edges, dtype=torch.long).t().contiguous()
        )
        train_edge_labels = torch.tensor(train_labels, dtype=torch.float)
        pos = int(train_edge_labels.sum())
        logger.info(
            f"Training: {len(train_labels):,} pairs "
            f"({pos:,} positive, {len(train_labels) - pos:,} negative)"
        )

        # Critical entity check
        for name, eid in [
            ("Adalimumab", self.adalimumab_id),
            ("Castleman", self.castleman_id),
            ("TNF", self.tnf_id),
        ]:
            if eid not in self.entity_to_idx:
                raise AssertionError(f"{name} ({eid}) not found in graph!")
        logger.info("Critical entities verified: Adalimumab, Castleman, TNF all present")

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            train_edge_index=train_edge_index,
            train_edge_labels=train_edge_labels,
            num_nodes=num_nodes,
        )

        logger.info("=" * 60)
        logger.info(
            f"Graph ready: {num_nodes:,} nodes, "
            f"{edge_index.shape[1]:,} directed edges"
        )
        logger.info("=" * 60)
        return data

    # ------------------------------------------------------------------
    # Edge weight generation
    # ------------------------------------------------------------------

    def get_disease_protein_neighbors(self, disease_id: str) -> List[str]:
        """
        Return all protein/gene neighbors of a disease node.
        Excludes drugs (CHEMBL.COMPOUND:) and other diseases (MONDO:).
        Must call build_graph() first to populate self.nx_graph.
        """
        if self.nx_graph is None:
            raise RuntimeError(
                "nx_graph is not loaded. Call build_graph() first."
            )
        if disease_id not in self.nx_graph:
            raise ValueError(f"Disease node {disease_id} not in graph.")

        neighbors = set()
        for n in list(self.nx_graph.successors(disease_id)) + list(
            self.nx_graph.predecessors(disease_id)
        ):
            if not n.startswith("CHEMBL.COMPOUND:") and not n.startswith("MONDO:"):
                neighbors.add(n)

        logger.info(
            f"Protein neighbors of {disease_id}: {len(neighbors)} found"
        )
        return list(neighbors)

    def make_random_weights(
        self,
        neighbors: List[str],
        seed: Optional[int] = None,
        weight_min: float = 0.5,
        weight_max: float = 5.0,
    ) -> Dict[str, float]:
        """
        Assign uniform-random weights to disease->protein edges.

        Purpose: Architecture validation. If drug rankings shift compared
        to uniform-weight baseline, GATConv attention is reading edge_attr.
        Once this is confirmed, we swap in real fold-change values.

        Args:
            neighbors:   List of protein node IDs (from get_disease_protein_neighbors)
            seed:        Random seed for reproducibility. None = random each call.
            weight_min:  Lower bound of uniform distribution (default 0.5)
            weight_max:  Upper bound of uniform distribution (default 5.0)
                         Range chosen to simulate realistic fold-change values.

        Returns:
            Dict mapping protein_id -> random float weight
        """
        if not neighbors:
            raise ValueError("neighbors list is empty. Nothing to weight.")

        rng = random.Random(seed)
        weights = {n: rng.uniform(weight_min, weight_max) for n in neighbors}

        logger.info(
            f"Random weights assigned to {len(weights)} neighbors "
            f"(seed={seed}, range=[{weight_min}, {weight_max}])"
        )
        if self.tnf_id in weights:
            logger.info(f"  TNF random weight: {weights[self.tnf_id]:.3f}")
        else:
            logger.warning(
                f"  TNF ({self.tnf_id}) not in neighbors list — "
                "it may not be a direct protein neighbor of Castleman"
            )

        actual_min = min(weights.values())
        actual_max = max(weights.values())
        logger.info(f"  Actual range: [{actual_min:.3f}, {actual_max:.3f}]")
        return weights

    def make_tnf_weights(
        self,
        neighbors: List[str],
        tnf_weight: Optional[float] = None,
        other_weight: float = 1.0,
    ) -> Dict[str, float]:
        """
        Set TNF->Castleman edge to fold-change weight, all others to 1.0.

        Used for real signal testing (once random weights validate architecture).
        The 30.7x value comes from iMCD-TAFRO scRNA-seq: TNF upregulation in
        naive CD4+ T cells. Will be replaced with cell-type-stratified values
        when Prof. Singh provides the full dataset.

        Args:
            neighbors:    List of protein node IDs
            tnf_weight:   Weight for TNF edge. Defaults to self.tnf_fold_change (30.7)
            other_weight: Weight for all non-TNF neighbors. Default 1.0.

        Returns:
            Dict mapping protein_id -> weight
        """
        if not neighbors:
            raise ValueError("neighbors list is empty. Nothing to weight.")

        if tnf_weight is None:
            tnf_weight = self.tnf_fold_change  # 30.7x

        weights = {n: other_weight for n in neighbors}

        if self.tnf_id in neighbors:
            weights[self.tnf_id] = tnf_weight
            logger.info(
                f"TNF edge weight: {tnf_weight:.2f}x "
                f"(all others: {other_weight:.1f})"
            )
        else:
            logger.warning(
                f"TNF ({self.tnf_id}) is NOT a direct protein neighbor of "
                "Castleman. Edge weight cannot be assigned. Check graph."
            )

        return weights

    # ------------------------------------------------------------------
    # Training and evaluation
    # ------------------------------------------------------------------

    def train_model(
        self,
        data: Data,
        model: GATModel,
        epochs: int = 200,
        lr: float = 0.01,
    ) -> GATModel:
        """
        Train GATConv model using RepoDB supervision labels (binary cross-entropy).
        Graph structure (edge_index + edge_attr) is used for message passing.
        Training pairs (train_edge_index) are used for supervision only.
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()
        model.train()
        logger.info(f"Training GATConv: {epochs} epochs, lr={lr}")

        for epoch in range(epochs):
            optimizer.zero_grad()
            z = model(data.x, data.edge_index, edge_attr=data.edge_attr)

            pos_mask = data.train_edge_labels == 1
            neg_mask = data.train_edge_labels == 0
            pos_pred = model.predict_links(z, data.train_edge_index[:, pos_mask])
            neg_pred = model.predict_links(z, data.train_edge_index[:, neg_mask])

            loss = criterion(
                pos_pred, torch.ones(pos_pred.size(0))
            ) + criterion(
                neg_pred, torch.zeros(neg_pred.size(0))
            )
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                logger.info(f"  Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

        logger.info("Training complete")
        return model

    def evaluate_drug_ranking(
        self, model: GATModel, data: Data, disease_entity: str
    ) -> Dict[str, float]:
        """
        Rank all 66,304 drugs by predicted link probability to a disease node.
        Returns OrderedDict sorted by score descending.
        """
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index, edge_attr=data.edge_attr)

            if disease_entity not in self.entity_to_idx:
                logger.warning(f"Disease entity not in graph: {disease_entity}")
                return {}

            disease_idx = self.entity_to_idx[disease_entity]
            drug_scores: Dict[str, float] = {}

            for entity, idx in self.entity_to_idx.items():
                if entity.startswith("CHEMBL.COMPOUND:"):
                    edge = torch.tensor([[idx], [disease_idx]], dtype=torch.long)
                    drug_scores[entity] = model.predict_links(z, edge).item()

        return dict(sorted(drug_scores.items(), key=lambda x: x[1], reverse=True))

    def get_adalimumab_rank(
        self, model: GATModel, data: Data, disease_entity: str
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Return (rank, score) of adalimumab for a given disease.
        Rank 1 = highest predicted affinity.
        Returns (None, None) if adalimumab not found in ranking.
        """
        ranking = self.evaluate_drug_ranking(model, data, disease_entity)
        for rank, (drug, score) in enumerate(ranking.items(), 1):
            if drug == self.adalimumab_id:
                return rank, score
        return None, None
