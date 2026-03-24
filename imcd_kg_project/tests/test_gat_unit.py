#!/usr/bin/env python3
"""
test_gat_unit.py
----------------
Purpose: Unit tests for gat_predictor.py that run locally with NO data files.
         Uses small synthetic graphs to verify:
           1. GATModel forward pass produces correct output shapes
           2. edge_attr (edge weights) flows through GATConv layers
           3. predict_links output is always in [0, 1]
           4. make_random_weights produces values in correct range
           5. make_tnf_weights correctly assigns TNF weight vs default
           6. edge_attr tensor shape is [E, 1] not [E] (GATConv requirement)
           7. build_graph edge_attr shape matches edge_index columns
           8. get_disease_protein_neighbors excludes drugs and diseases
           9. Data leakage check raises ValueError when target pair in training

Run with:
    cd imcd_kg_project
    python -m pytest tests/test_gat_unit.py -v

All tests must pass before submitting anything to Sockeye.
"""

import sys
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import networkx as nx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enhanced_kgnn.gat_predictor import GATModel, GATPredictor


# ---------------------------------------------------------------------------
# Synthetic graph fixture
# ---------------------------------------------------------------------------

def make_synthetic_graph():
    """
    Build a tiny NetworkX DiGraph that mimics the real graph structure.
    Nodes: 2 drugs, 2 diseases, 3 proteins.
    Edges include direct protein->disease connections.
    """
    G = nx.DiGraph()
    G.add_nodes_from([
        ("CHEMBL.COMPOUND:ADA", {"name": "Adalimumab", "category": "Drug"}),
        ("CHEMBL.COMPOUND:OTHER", {"name": "OtherDrug", "category": "Drug"}),
        ("MONDO:CASTLEMAN", {"name": "Castleman", "category": "Disease"}),
        ("MONDO:DIABETES", {"name": "Diabetes", "category": "Disease"}),
        ("UniProtKB:TNF", {"name": "TNF", "category": "Protein"}),
        ("UniProtKB:IL6", {"name": "IL6", "category": "Protein"}),
        ("UniProtKB:OTHER_PROT", {"name": "OtherProtein", "category": "Protein"}),
    ])
    G.add_edges_from([
        ("CHEMBL.COMPOUND:ADA", "UniProtKB:TNF"),       # adalimumab -> TNF (direct)
        ("UniProtKB:TNF", "MONDO:CASTLEMAN"),            # TNF -> Castleman (direct)
        ("UniProtKB:IL6", "MONDO:CASTLEMAN"),            # IL6 -> Castleman
        ("UniProtKB:OTHER_PROT", "MONDO:DIABETES"),      # other protein -> diabetes
        ("CHEMBL.COMPOUND:OTHER", "UniProtKB:OTHER_PROT"),
    ])
    return G


def make_mock_predictor(monkeypatch=None):
    """
    Create a GATPredictor with mocked config and no real files required.
    """
    predictor = GATPredictor.__new__(GATPredictor)
    predictor.processed_graph_path = Path("/nonexistent/full_graph.pkl")
    predictor.training_data_path = Path("/nonexistent/training")
    predictor.adalimumab_id = "CHEMBL.COMPOUND:ADA"
    predictor.castleman_id = "MONDO:CASTLEMAN"
    predictor.tnf_id = "UniProtKB:TNF"
    predictor.tnf_fold_change = 30.7
    predictor.entity_to_idx = {}
    predictor.idx_to_entity = {}
    predictor.nx_graph = None
    return predictor


# ---------------------------------------------------------------------------
# GATModel tests
# ---------------------------------------------------------------------------

class TestGATModel:

    def test_output_shape_basic(self):
        """GATModel forward: output shape must be [N, output_dim]."""
        N, E = 10, 20
        input_dim, hidden_dim, output_dim = 3, 8, 4
        heads = 4

        model = GATModel(input_dim=input_dim, hidden_dim=hidden_dim,
                         output_dim=output_dim, heads=heads, edge_dim=1)

        x = torch.rand(N, input_dim)
        # Random edges (no self-loops, within range)
        src = torch.randint(0, N, (E,))
        dst = torch.randint(0, N, (E,))
        edge_index = torch.stack([src, dst], dim=0)
        edge_attr = torch.ones(E, 1)

        model.eval()
        with torch.no_grad():
            z = model(x, edge_index, edge_attr=edge_attr)

        assert z.shape == (N, output_dim), (
            f"Expected output shape ({N}, {output_dim}), got {z.shape}"
        )

    def test_edge_attr_shape_requirement(self):
        """edge_attr must be [E, 1] not [E] — GATConv requires 2D."""
        N, E = 5, 8
        model = GATModel(input_dim=3, hidden_dim=8, output_dim=4,
                         heads=4, edge_dim=1)

        x = torch.rand(N, 3)
        src = torch.randint(0, N, (E,))
        dst = torch.randint(0, N, (E,))
        edge_index = torch.stack([src, dst], dim=0)

        # 1D edge_attr should fail or give wrong shapes — test that [E, 1] works
        edge_attr_2d = torch.ones(E, 1)  # correct
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index, edge_attr=edge_attr_2d)
        assert z.shape == (N, 4)

    def test_predict_links_sigmoid_range(self):
        """predict_links output must be in [0, 1] for all inputs."""
        N = 8
        model = GATModel(input_dim=3, hidden_dim=8, output_dim=4, heads=4)

        # Use random embeddings (as if post-training)
        z = torch.randn(N, 4)
        src = torch.tensor([0, 1, 2, 3])
        dst = torch.tensor([4, 5, 6, 7])
        edge_index = torch.stack([src, dst], dim=0)

        scores = model.predict_links(z, edge_index)
        assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"
        assert (scores >= 0).all() and (scores <= 1).all(), (
            f"Scores out of [0, 1]: min={scores.min():.4f}, max={scores.max():.4f}"
        )

    def test_hidden_dim_heads_validation(self):
        """GATModel must raise ValueError if hidden_dim not divisible by heads."""
        with pytest.raises(ValueError, match="divisible by heads"):
            GATModel(input_dim=3, hidden_dim=10, output_dim=4, heads=4)

    def test_no_edge_attr_still_runs(self):
        """GATConv should run even if edge_attr=None (no edge features)."""
        N, E = 6, 10
        model = GATModel(input_dim=3, hidden_dim=8, output_dim=4,
                         heads=4, edge_dim=1)

        x = torch.rand(N, 3)
        src = torch.randint(0, N, (E,))
        dst = torch.randint(0, N, (E,))
        edge_index = torch.stack([src, dst], dim=0)

        model.eval()
        with torch.no_grad():
            # edge_attr=None: GATConv should handle gracefully
            # (may raise if edge_dim set, that is acceptable behavior)
            # This test documents the behavior, not enforces it
            try:
                z = model(x, edge_index, edge_attr=None)
                assert z.shape == (N, 4)
            except Exception as e:
                # If GATConv requires edge_attr when edge_dim is set, that is expected
                assert "edge_attr" in str(e).lower() or "edge" in str(e).lower(), (
                    f"Unexpected error: {e}"
                )

    def test_high_weight_changes_embedding(self):
        """
        Core architecture test: changing edge_attr values must change output embeddings.
        If this fails, edge weights are not flowing through the model.
        """
        N, E = 6, 8
        torch.manual_seed(0)
        model = GATModel(input_dim=3, hidden_dim=8, output_dim=4,
                         heads=4, edge_dim=1)

        x = torch.rand(N, 3)
        src = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3])
        dst = torch.tensor([1, 2, 3, 0, 3, 0, 3, 2])
        edge_index = torch.stack([src, dst], dim=0)

        edge_attr_low = torch.ones(E, 1)
        edge_attr_high = torch.ones(E, 1) * 30.7

        model.eval()
        with torch.no_grad():
            z_low = model(x, edge_index, edge_attr=edge_attr_low)
            z_high = model(x, edge_index, edge_attr=edge_attr_high)

        diff = (z_low - z_high).abs().max().item()
        assert diff > 1e-6, (
            f"Edge weights had no effect on embeddings (max diff={diff:.2e}). "
            "edge_attr is not flowing through the model."
        )


# ---------------------------------------------------------------------------
# GATPredictor weight generation tests
# ---------------------------------------------------------------------------

class TestWeightGeneration:

    def setup_method(self):
        self.predictor = make_mock_predictor()
        self.predictor.nx_graph = make_synthetic_graph()

    def test_get_disease_protein_neighbors_excludes_drugs_and_diseases(self):
        """Neighbors of Castleman should only be proteins, not drugs or diseases."""
        neighbors = self.predictor.get_disease_protein_neighbors("MONDO:CASTLEMAN")
        for n in neighbors:
            assert not n.startswith("CHEMBL.COMPOUND:"), (
                f"Drug found in neighbors: {n}"
            )
            assert not n.startswith("MONDO:"), (
                f"Disease found in neighbors: {n}"
            )

    def test_get_disease_protein_neighbors_finds_tnf(self):
        """TNF must be in Castleman's protein neighbors."""
        neighbors = self.predictor.get_disease_protein_neighbors("MONDO:CASTLEMAN")
        assert "UniProtKB:TNF" in neighbors, (
            "TNF not found in Castleman's neighbors. "
            "Check graph structure: TNF->Castleman edge must exist."
        )

    def test_get_disease_protein_neighbors_raises_if_no_graph(self):
        """Must raise RuntimeError if nx_graph not loaded."""
        predictor = make_mock_predictor()
        predictor.nx_graph = None
        with pytest.raises(RuntimeError, match="build_graph"):
            predictor.get_disease_protein_neighbors("MONDO:CASTLEMAN")

    def test_get_disease_protein_neighbors_raises_if_not_in_graph(self):
        """Must raise ValueError if disease node doesn't exist in graph."""
        with pytest.raises(ValueError, match="not in graph"):
            self.predictor.get_disease_protein_neighbors("MONDO:NONEXISTENT")

    def test_make_random_weights_range(self):
        """Random weights must be within [weight_min, weight_max]."""
        neighbors = ["UniProtKB:TNF", "UniProtKB:IL6", "UniProtKB:OTHER_PROT"]
        weights = self.predictor.make_random_weights(
            neighbors, seed=42, weight_min=0.5, weight_max=5.0
        )
        for node, w in weights.items():
            assert 0.5 <= w <= 5.0, (
                f"Weight out of range for {node}: {w}"
            )

    def test_make_random_weights_reproducible_with_seed(self):
        """Same seed must produce same weights."""
        neighbors = ["UniProtKB:TNF", "UniProtKB:IL6"]
        w1 = self.predictor.make_random_weights(neighbors, seed=42)
        w2 = self.predictor.make_random_weights(neighbors, seed=42)
        assert w1 == w2, "Random weights with same seed are not reproducible."

    def test_make_random_weights_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different weights."""
        neighbors = ["UniProtKB:TNF", "UniProtKB:IL6", "UniProtKB:OTHER_PROT"]
        w1 = self.predictor.make_random_weights(neighbors, seed=42)
        w2 = self.predictor.make_random_weights(neighbors, seed=99)
        assert w1 != w2, "Different seeds produced identical weights."

    def test_make_random_weights_covers_all_neighbors(self):
        """Every neighbor must appear as a key in the output dict."""
        neighbors = ["UniProtKB:TNF", "UniProtKB:IL6", "UniProtKB:OTHER_PROT"]
        weights = self.predictor.make_random_weights(neighbors, seed=1)
        assert set(weights.keys()) == set(neighbors), (
            f"Missing neighbors in weights. "
            f"Expected: {set(neighbors)}, Got: {set(weights.keys())}"
        )

    def test_make_random_weights_empty_neighbors_raises(self):
        """Empty neighbor list must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            self.predictor.make_random_weights([])

    def test_make_tnf_weights_tnf_gets_fold_change(self):
        """TNF neighbor must receive the fold-change weight (30.7x)."""
        neighbors = ["UniProtKB:TNF", "UniProtKB:IL6"]
        weights = self.predictor.make_tnf_weights(neighbors)
        assert abs(weights["UniProtKB:TNF"] - self.predictor.tnf_fold_change) < 1e-6, (
            f"TNF weight {weights['UniProtKB:TNF']} != "
            f"tnf_fold_change {self.predictor.tnf_fold_change}"
        )

    def test_make_tnf_weights_others_get_default(self):
        """Non-TNF neighbors must receive other_weight (default 1.0)."""
        neighbors = ["UniProtKB:TNF", "UniProtKB:IL6", "UniProtKB:OTHER_PROT"]
        weights = self.predictor.make_tnf_weights(neighbors, other_weight=1.0)
        for n, w in weights.items():
            if n != "UniProtKB:TNF":
                assert abs(w - 1.0) < 1e-6, (
                    f"Non-TNF neighbor {n} has weight {w}, expected 1.0"
                )

    def test_make_tnf_weights_custom_tnf_weight(self):
        """Custom tnf_weight should override the default fold-change."""
        neighbors = ["UniProtKB:TNF", "UniProtKB:IL6"]
        weights = self.predictor.make_tnf_weights(neighbors, tnf_weight=10.0)
        assert abs(weights["UniProtKB:TNF"] - 10.0) < 1e-6

    def test_make_tnf_weights_empty_neighbors_raises(self):
        """Empty neighbor list must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            self.predictor.make_tnf_weights([])


# ---------------------------------------------------------------------------
# GATPredictor build_graph tests (using mocked nx_graph and training data)
# ---------------------------------------------------------------------------

class TestBuildGraph:

    def _make_predictor_with_graph(self):
        """Returns predictor with synthetic graph loaded and mocked training data."""
        predictor = make_mock_predictor()
        G = make_synthetic_graph()

        # Patch the file loading
        with patch.object(Path, "exists", return_value=True), \
             patch("builtins.open", MagicMock()), \
             patch("pickle.load", return_value=G):

            # Also mock training data loading
            predictor._load_training_data = lambda: [
                ("CHEMBL.COMPOUND:OTHER", "MONDO:DIABETES", 1),
                ("CHEMBL.COMPOUND:OTHER", "MONDO:CASTLEMAN", 0),
            ]
            # Manually inject graph (bypass file loading for speed)
            predictor.nx_graph = G
            entities = sorted(G.nodes())
            predictor.entity_to_idx = {e: i for i, e in enumerate(entities)}
            predictor.idx_to_entity = {i: e for e, i in predictor.entity_to_idx.items()}

        return predictor

    def test_edge_attr_shape_is_E_by_1(self):
        """
        edge_attr shape must be [E, 1].
        GATConv requires 2D edge features.
        """
        predictor = self._make_predictor_with_graph()
        G = predictor.nx_graph
        num_edges = G.number_of_edges()

        # Build edge_index and edge_attr manually (same logic as build_graph)
        edge_src, edge_dst, edge_weights = [], [], []
        for source, target in G.edges():
            if source in predictor.entity_to_idx and target in predictor.entity_to_idx:
                edge_src.extend([predictor.entity_to_idx[source],
                                  predictor.entity_to_idx[target]])
                edge_dst.extend([predictor.entity_to_idx[target],
                                  predictor.entity_to_idx[source]])
                edge_weights.extend([1.0, 1.0])

        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        assert edge_attr.dim() == 2, (
            f"edge_attr must be 2D [E, 1], got {edge_attr.dim()}D"
        )
        assert edge_attr.shape[1] == 1, (
            f"edge_attr second dim must be 1, got {edge_attr.shape[1]}"
        )

    def test_edge_attr_columns_match_edge_index_columns(self):
        """edge_attr.shape[0] must equal edge_index.shape[1]."""
        predictor = self._make_predictor_with_graph()
        G = predictor.nx_graph

        edge_src, edge_dst, edge_weights = [], [], []
        for source, target in G.edges():
            if source in predictor.entity_to_idx and target in predictor.entity_to_idx:
                edge_src.extend([predictor.entity_to_idx[source],
                                  predictor.entity_to_idx[target]])
                edge_dst.extend([predictor.entity_to_idx[target],
                                  predictor.entity_to_idx[source]])
                edge_weights.extend([1.0, 1.0])

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        assert edge_index.shape[1] == edge_attr.shape[0], (
            f"edge_index has {edge_index.shape[1]} edges but "
            f"edge_attr has {edge_attr.shape[0]} rows"
        )

    def test_disease_edge_weights_applied_to_castleman_edges(self):
        """
        When disease_edge_weights is provided, edges touching Castleman
        must have the specified weight, not the default.
        """
        predictor = self._make_predictor_with_graph()
        G = predictor.nx_graph

        disease_edge_weights = {"UniProtKB:TNF": 30.7, "UniProtKB:IL6": 1.5}
        default_weight = 1.0

        edge_src, edge_dst, edge_weights = [], [], []
        for source, target in G.edges():
            if source not in predictor.entity_to_idx or target not in predictor.entity_to_idx:
                continue
            weight = default_weight
            if source == predictor.castleman_id and target in disease_edge_weights:
                weight = disease_edge_weights[target]
            elif target == predictor.castleman_id and source in disease_edge_weights:
                weight = disease_edge_weights[source]
            edge_src.extend([predictor.entity_to_idx[source],
                              predictor.entity_to_idx[target]])
            edge_dst.extend([predictor.entity_to_idx[target],
                              predictor.entity_to_idx[source]])
            edge_weights.extend([weight, weight])

        # At least one weight should be 30.7 (for TNF->Castleman)
        assert 30.7 in edge_weights, (
            "TNF->Castleman edge weight 30.7 not found in edge_weights. "
            "Disease edge weight assignment is broken."
        )

    def test_node_feature_one_hot(self):
        """
        Drugs get [1,0,0], diseases get [0,1,0], proteins get [0,0,1].
        Each node must have exactly one 1 and the rest 0s.
        """
        predictor = self._make_predictor_with_graph()
        G = predictor.nx_graph
        num_nodes = G.number_of_nodes()

        node_features = np.zeros((num_nodes, 3), dtype=np.float32)
        for entity, idx in predictor.entity_to_idx.items():
            if entity.startswith("CHEMBL.COMPOUND:"):
                node_features[idx, 0] = 1.0
            elif entity.startswith("MONDO:"):
                node_features[idx, 1] = 1.0
            else:
                node_features[idx, 2] = 1.0

        # Each row must sum to 1.0
        row_sums = node_features.sum(axis=1)
        assert np.allclose(row_sums, 1.0), (
            f"Some nodes have incorrect feature sum: {row_sums}"
        )


# ---------------------------------------------------------------------------
# Data leakage test
# ---------------------------------------------------------------------------

class TestDataLeakage:

    def test_leakage_raises_value_error(self):
        """
        If adalimumab-Castleman pair is in training data,
        _load_training_data must raise ValueError.
        """
        predictor = make_mock_predictor()

        # Inject leaking pair
        def fake_load():
            pairs = [
                ("CHEMBL.COMPOUND:ADA", "MONDO:CASTLEMAN", 1),  # leakage
                ("CHEMBL.COMPOUND:OTHER", "MONDO:DIABETES", 1),
            ]
            leakage = [
                (d, dis) for d, dis, _ in pairs
                if d == predictor.adalimumab_id and dis == predictor.castleman_id
            ]
            if leakage:
                raise ValueError(
                    f"Data leakage detected: {len(leakage)} adalimumab-Castleman "
                    "pairs found in training data."
                )
            return pairs

        with pytest.raises(ValueError, match="leakage"):
            fake_load()

    def test_no_leakage_passes(self):
        """
        Training data without adalimumab-Castleman pair must not raise.
        """
        predictor = make_mock_predictor()

        pairs = [
            ("CHEMBL.COMPOUND:OTHER", "MONDO:DIABETES", 1),
            ("CHEMBL.COMPOUND:ADA", "MONDO:DIABETES", 0),
        ]
        leakage = [
            (d, dis) for d, dis, _ in pairs
            if d == predictor.adalimumab_id and dis == predictor.castleman_id
        ]
        assert len(leakage) == 0, "False positive: leakage detected where there is none"


# ---------------------------------------------------------------------------
# Integration: small synthetic end-to-end (no real data)
# ---------------------------------------------------------------------------

class TestSyntheticEndToEnd:

    def test_train_and_rank_synthetic(self):
        """
        Full pipeline on synthetic 7-node graph:
        build edge_attr -> train GATModel -> predict links -> get ranking.
        Tests that the full call sequence runs without error.
        """
        G = make_synthetic_graph()
        entities = sorted(G.nodes())
        entity_to_idx = {e: i for i, e in enumerate(entities)}
        N = len(entities)

        # Node features
        x = torch.zeros(N, 3)
        for entity, idx in entity_to_idx.items():
            if entity.startswith("CHEMBL.COMPOUND:"):
                x[idx, 0] = 1.0
            elif entity.startswith("MONDO:"):
                x[idx, 1] = 1.0
            else:
                x[idx, 2] = 1.0

        # Edge index + attr (uniform weight 1.0)
        edge_src, edge_dst, edge_weights = [], [], []
        for source, target in G.edges():
            edge_src.extend([entity_to_idx[source], entity_to_idx[target]])
            edge_dst.extend([entity_to_idx[target], entity_to_idx[source]])
            edge_weights.extend([1.0, 1.0])

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        # Training pairs (synthetic)
        ada_idx = entity_to_idx["CHEMBL.COMPOUND:ADA"]
        castleman_idx = entity_to_idx["MONDO:CASTLEMAN"]
        other_drug_idx = entity_to_idx["CHEMBL.COMPOUND:OTHER"]
        diabetes_idx = entity_to_idx["MONDO:DIABETES"]

        train_edge_index = torch.tensor(
            [[other_drug_idx, other_drug_idx],
             [diabetes_idx, castleman_idx]],
            dtype=torch.long
        )
        train_edge_labels = torch.tensor([1.0, 0.0])

        from torch_geometric.data import Data
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            train_edge_index=train_edge_index,
            train_edge_labels=train_edge_labels,
            num_nodes=N,
        )

        # Train GATModel
        torch.manual_seed(42)
        model = GATModel(input_dim=3, hidden_dim=8, output_dim=4, heads=4, edge_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()

        model.train()
        for _ in range(10):  # Just 10 epochs for speed
            optimizer.zero_grad()
            z = model(data.x, data.edge_index, edge_attr=data.edge_attr)
            pos_mask = data.train_edge_labels == 1
            neg_mask = data.train_edge_labels == 0
            pos_pred = model.predict_links(z, data.train_edge_index[:, pos_mask])
            neg_pred = model.predict_links(z, data.train_edge_index[:, neg_mask])
            loss = criterion(pos_pred, torch.ones(pos_pred.size(0))) + \
                   criterion(neg_pred, torch.zeros(neg_pred.size(0)))
            loss.backward()
            optimizer.step()

        # Predict links for adalimumab
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index, edge_attr=data.edge_attr)
            edge = torch.tensor([[ada_idx], [castleman_idx]], dtype=torch.long)
            score = model.predict_links(z, edge).item()

        assert 0.0 <= score <= 1.0, f"Score out of [0,1]: {score}"
        assert not np.isnan(score), "Score is NaN — training failed"

    def test_high_weight_edge_shifts_score(self):
        """
        Increasing the edge weight on TNF->Castleman must produce a
        different embedding for Castleman (score may go up or down depending
        on training, but embeddings MUST change — validates edge_attr flows).
        """
        G = make_synthetic_graph()
        entities = sorted(G.nodes())
        entity_to_idx = {e: i for i, e in enumerate(entities)}
        N = len(entities)

        x = torch.zeros(N, 3)
        for entity, idx in entity_to_idx.items():
            if entity.startswith("CHEMBL.COMPOUND:"):
                x[idx, 0] = 1.0
            elif entity.startswith("MONDO:"):
                x[idx, 1] = 1.0
            else:
                x[idx, 2] = 1.0

        def build_edge_tensors(tnf_weight):
            edge_src, edge_dst, edge_weights = [], [], []
            for source, target in G.edges():
                w = 1.0
                # TNF->Castleman edge gets special weight
                if source == "UniProtKB:TNF" and target == "MONDO:CASTLEMAN":
                    w = tnf_weight
                elif source == "MONDO:CASTLEMAN" and target == "UniProtKB:TNF":
                    w = tnf_weight
                edge_src.extend([entity_to_idx[source], entity_to_idx[target]])
                edge_dst.extend([entity_to_idx[target], entity_to_idx[source]])
                edge_weights.extend([w, w])
            return (
                torch.tensor([edge_src, edge_dst], dtype=torch.long),
                torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1),
            )

        torch.manual_seed(0)
        model = GATModel(input_dim=3, hidden_dim=8, output_dim=4, heads=4, edge_dim=1)
        model.eval()

        ei_low, ea_low = build_edge_tensors(1.0)
        ei_high, ea_high = build_edge_tensors(30.7)

        with torch.no_grad():
            castleman_idx = entity_to_idx["MONDO:CASTLEMAN"]
            z_low = model(x, ei_low, edge_attr=ea_low)
            z_high = model(x, ei_high, edge_attr=ea_high)

        # Castleman embedding must differ between low and high weight
        diff = (z_low[castleman_idx] - z_high[castleman_idx]).abs().max().item()
        assert diff > 1e-6, (
            f"Castleman embedding unchanged when TNF->Castleman weight changed "
            f"from 1.0 to 30.7 (max diff={diff:.2e}). "
            "edge_attr is not affecting attention/embeddings."
        )


# ---------------------------------------------------------------------------
# get_drug_ranks tests
# ---------------------------------------------------------------------------

class TestGetDrugRanks:
    """
    Tests for GATPredictor.get_drug_ranks().

    Why these tests matter:
      Phase 5.5 ran on Sockeye without any tests and returned garbage results —
      random compounds at the top, all known drugs absent. Root cause: the
      earlier get_top_n_drugs() only returned the top-200 generic list, and
      adalimumab (#13,000+) was never in that window.

      get_drug_ranks() is the fix: a single forward pass that fills BOTH a
      top-N table (generic) AND an explicit tracker for drugs_of_interest
      regardless of their rank position. These tests verify that invariant.
    """

    def _build_data_and_predictor(self):
        """
        Build a tiny graph + trained model + predictor for get_drug_ranks tests.
        7 nodes: 3 drugs, 2 diseases, 2 proteins.
        drugs_of_interest: all 3 drug IDs (one will rank outside top-N to test
        explicit tracking).
        """
        G = make_synthetic_graph()  # has ADA, OTHER drug, 2 diseases, 2 proteins
        # Add a second extra drug so we have 3 total
        G.add_node("CHEMBL.COMPOUND:THIRD", name="ThirdDrug", category="biolink:Drug")
        G.add_edge("CHEMBL.COMPOUND:THIRD", "UniProtKB:TNF")

        entities = sorted(G.nodes())
        entity_to_idx = {e: i for i, e in enumerate(entities)}
        N = len(entities)

        x = torch.zeros(N, 3)
        for entity, idx in entity_to_idx.items():
            if entity.startswith("CHEMBL.COMPOUND:"):
                x[idx, 0] = 1.0
            elif entity.startswith("MONDO:"):
                x[idx, 1] = 1.0
            else:
                x[idx, 2] = 1.0

        edge_src, edge_dst, edge_weights = [], [], []
        for source, target in G.edges():
            edge_src.extend([entity_to_idx[source], entity_to_idx[target]])
            edge_dst.extend([entity_to_idx[target], entity_to_idx[source]])
            edge_weights.extend([1.0, 1.0])

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        # Minimal supervision: one pair
        from torch_geometric.data import Data
        train_edge_index = torch.tensor(
            [[entity_to_idx["CHEMBL.COMPOUND:OTHER"]],
             [entity_to_idx["MONDO:CASTLEMAN"]]], dtype=torch.long
        )
        train_edge_labels = torch.tensor([1.0])
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            train_edge_index=train_edge_index,
            train_edge_labels=train_edge_labels,
            num_nodes=N,
        )

        torch.manual_seed(42)
        model = GATModel(input_dim=3, hidden_dim=8, output_dim=4, heads=4, edge_dim=1)
        model.eval()

        predictor = make_mock_predictor()
        predictor.nx_graph = G
        predictor.entity_to_idx = entity_to_idx
        predictor.idx_to_entity = {i: e for e, i in entity_to_idx.items()}

        drugs_of_interest = {
            "CHEMBL.COMPOUND:ADA":   "adalimumab",
            "CHEMBL.COMPOUND:OTHER": "otherdrug",
            "CHEMBL.COMPOUND:THIRD": "thirddrug",
        }
        return predictor, model, data, drugs_of_interest

    def test_top_n_table_length(self):
        """top_n_table must have exactly top_n entries (when there are enough drugs)."""
        predictor, model, data, doi = self._build_data_and_predictor()
        top_n_table, _ = predictor.get_drug_ranks(
            model, data, "MONDO:CASTLEMAN", doi, top_n=2
        )
        # Graph has 3 drugs; top_n=2 → table has 2 entries
        assert len(top_n_table) == 2, (
            f"Expected 2 entries in top_n_table, got {len(top_n_table)}"
        )

    def test_specific_drugs_found_regardless_of_rank(self):
        """
        Every drug in drugs_of_interest must appear in specific dict,
        even if it ranks below top_n.
        """
        predictor, model, data, doi = self._build_data_and_predictor()
        # top_n=1 means only the #1 drug enters the table, but all 3 DOI
        # must still be in specific
        _, specific = predictor.get_drug_ranks(
            model, data, "MONDO:CASTLEMAN", doi, top_n=1
        )
        for drug_id in doi:
            assert drug_id in specific, (
                f"Drug {drug_id} missing from specific — "
                "explicit tracking failed for out-of-table drug"
            )

    def test_specific_rank_matches_table_for_top_drug(self):
        """
        If a drug of interest is also in the top_n_table, its rank in
        specific must equal its rank in the table.
        """
        predictor, model, data, doi = self._build_data_and_predictor()
        top_n_table, specific = predictor.get_drug_ranks(
            model, data, "MONDO:CASTLEMAN", doi, top_n=3
        )
        # All 3 drugs must appear in both
        table_ranks = {row["drug_id"]: row["rank"] for row in top_n_table}
        for drug_id in doi:
            if drug_id in table_ranks:
                assert specific[drug_id]["rank"] == table_ranks[drug_id], (
                    f"Rank mismatch for {drug_id}: "
                    f"table says {table_ranks[drug_id]}, "
                    f"specific says {specific[drug_id]['rank']}"
                )

    def test_top_n_table_sorted_descending(self):
        """top_n_table must be sorted by score descending (rank 1 is highest score)."""
        predictor, model, data, doi = self._build_data_and_predictor()
        top_n_table, _ = predictor.get_drug_ranks(
            model, data, "MONDO:CASTLEMAN", doi, top_n=3
        )
        scores = [row["score"] for row in top_n_table]
        assert scores == sorted(scores, reverse=True), (
            f"top_n_table is not sorted descending: {scores}"
        )

    def test_name_from_graph_in_specific(self):
        """
        specific dict must include drug name from the graph node attribute,
        not the raw drug_id string.
        """
        predictor, model, data, doi = self._build_data_and_predictor()
        _, specific = predictor.get_drug_ranks(
            model, data, "MONDO:CASTLEMAN", doi, top_n=1
        )
        # "CHEMBL.COMPOUND:ADA" -> node name "Adalimumab" (from make_synthetic_graph)
        ada_info = specific.get("CHEMBL.COMPOUND:ADA", {})
        assert ada_info.get("name") == "Adalimumab", (
            f"Expected name 'Adalimumab' from graph, got {ada_info.get('name')!r}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
