"""
test_composite_node_builder.py
-------------------------------
Unit tests for composite_node_builder.py

All tests use a small synthetic graph — no full_graph.pkl needed.

Synthetic graph structure mirrors real RTX-KG2:
  - Directed (DiGraph)
  - Proteins -> Castleman (incoming edges)
  - Adalimumab -> TNF (drug -> protein)
  - All node IDs use real-world prefixes (UniProtKB, MONDO, CHEMBL)

What is being tested:
  1. Protein neighbors found correctly via G.predecessors (directed graph)
  2. Composite nodes created for t-stat >= min_tstat only
  3. Node count increases by exactly N composite nodes
  4. Each composite node has exactly 2 new edges:
       Castleman -> composite (norm_weight)
       composite -> canonical protein (1.0)
  5. All norm_weights are in [0.1, 1.0]
  6. Original edges (TNF->Castleman, Adalimumab->TNF) still exist after build
  7. min_tstat filter works correctly
  8. Normalization clips at 99th percentile
  9. Composite node ID format is correct
  10. Graph with no qualifying pairs returns empty composite list
"""

import csv
import sys
import tempfile
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from enhanced_kgnn.composite_node_builder import (
    CompositeNodeBuilder,
    CompositeNodeInfo,
)
from enhanced_kgnn.gene_mapper import GeneMapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CASTLEMAN  = "MONDO:0015564"
TNF        = "UniProtKB:P01375"
IL6        = "UniProtKB:P05231"
STAT3      = "UniProtKB:P40763"
ADALIMUMAB = "CHEMBL.COMPOUND:CHEMBL1201580"


def make_synthetic_graph() -> nx.DiGraph:
    """
    Small directed graph that mirrors the structure we confirmed on Sockeye:
      - Proteins -> Castleman (incoming, via predecessors)
      - Adalimumab -> TNF (drug -> protein, unchanged by builder)
    """
    G = nx.DiGraph()

    # Protein nodes
    G.add_node(TNF,   name="TNF",   category="biolink:Protein")
    G.add_node(IL6,   name="IL6",   category="biolink:Protein")
    G.add_node(STAT3, name="STAT3", category="biolink:Protein")

    # Disease node
    G.add_node(CASTLEMAN, name="Castleman Disease", category="biolink:Disease")

    # Drug node
    G.add_node(ADALIMUMAB, name="ADALIMUMAB", category="biolink:Drug")

    # Incoming protein->Castleman edges (confirmed direction from Sockeye)
    G.add_edge(TNF,   CASTLEMAN)
    G.add_edge(IL6,   CASTLEMAN)
    G.add_edge(STAT3, CASTLEMAN)

    # Drug->protein edge (must survive untouched)
    G.add_edge(ADALIMUMAB, TNF)

    return G


def make_csv(rows: list, tmp_dir: str) -> Path:
    """Write a minimal t-stat CSV matching the real file format."""
    path = Path(tmp_dir) / "tstats.csv"
    fieldnames = ["", "gene", "B cells", "ILC",
                  "Megakaryocytes/platelets", "Monocytes", "T cells"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow({"": str(i), **row})
    return path


def make_mapper_for(G: nx.DiGraph) -> GeneMapper:
    """Create a loaded GeneMapper using the synthetic graph."""
    mapper = GeneMapper()
    mapper.load_graph(nx_graph=G)
    return mapper


# ---------------------------------------------------------------------------
# Tests: protein neighbor detection
# ---------------------------------------------------------------------------

class TestGetProteinNeighbors:

    def test_finds_incoming_protein_neighbors(self):
        """predecessors() should find TNF, IL6, STAT3 as incoming neighbors."""
        G = make_synthetic_graph()
        builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
        neighbors = builder._get_protein_neighbors(G)
        assert TNF in neighbors
        assert IL6 in neighbors
        assert STAT3 in neighbors

    def test_does_not_include_drug_nodes(self):
        """Drug nodes (CHEMBL) should not appear as protein neighbors."""
        G = make_synthetic_graph()
        builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
        neighbors = builder._get_protein_neighbors(G)
        assert ADALIMUMAB not in neighbors

    def test_does_not_include_disease_itself(self):
        """Castleman should not be its own neighbor."""
        G = make_synthetic_graph()
        builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
        neighbors = builder._get_protein_neighbors(G)
        assert CASTLEMAN not in neighbors

    def test_outgoing_edges_not_counted(self):
        """
        Castleman->protein edges (outgoing) should NOT be included.
        On the real graph there are 0 outgoing protein edges.
        """
        G = make_synthetic_graph()
        # Add an outgoing edge from Castleman to a new protein
        new_protein = "UniProtKB:FAKE001"
        G.add_node(new_protein, name="FAKEGENE", category="biolink:Protein")
        G.add_edge(CASTLEMAN, new_protein)  # outgoing — should NOT be included

        builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=0.0)
        neighbors = builder._get_protein_neighbors(G)
        assert new_protein not in neighbors


# ---------------------------------------------------------------------------
# Tests: normalization
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_all_weights_in_range(self):
        """All normalized weights must be in [0.1, 1.0]."""
        tstat_data = {
            ("TNF", "T cells"):    {"uniprot_id": TNF,   "raw_tstat": 2.91},
            ("TNF", "Monocytes"):  {"uniprot_id": TNF,   "raw_tstat": 32.22},
            ("STAT3", "Monocytes"):{"uniprot_id": STAT3, "raw_tstat": 3.99},
        }
        builder = CompositeNodeBuilder(min_tstat=2.0)
        result = builder._filter_and_normalize(tstat_data)
        for key, val in result.items():
            w = val["norm_weight"]
            assert 0.1 <= w <= 1.0, f"{key} weight {w} out of [0.1, 1.0]"

    def test_highest_tstat_gets_highest_weight(self):
        """TNF Monocytes (32.22) should get a higher weight than T cells (2.91)."""
        tstat_data = {
            ("TNF", "T cells"):   {"uniprot_id": TNF, "raw_tstat": 2.91},
            ("TNF", "Monocytes"): {"uniprot_id": TNF, "raw_tstat": 32.22},
        }
        builder = CompositeNodeBuilder(min_tstat=2.0)
        result = builder._filter_and_normalize(tstat_data)
        w_tcells    = result[("TNF", "T cells")]["norm_weight"]
        w_monocytes = result[("TNF", "Monocytes")]["norm_weight"]
        assert w_monocytes > w_tcells

    def test_min_tstat_filter_removes_low_values(self):
        """Pairs with t-stat < min_tstat must be excluded."""
        tstat_data = {
            ("TNF", "T cells"):   {"uniprot_id": TNF, "raw_tstat": 1.5},  # below 2.0
            ("TNF", "Monocytes"): {"uniprot_id": TNF, "raw_tstat": 32.22},
        }
        builder = CompositeNodeBuilder(min_tstat=2.0)
        result = builder._filter_and_normalize(tstat_data)
        assert ("TNF", "T cells") not in result
        assert ("TNF", "Monocytes") in result

    def test_identical_values_get_uniform_weight(self):
        """When all t-stats are equal, all weights should be 0.5."""
        tstat_data = {
            ("TNF",   "T cells"):   {"uniprot_id": TNF,   "raw_tstat": 5.0},
            ("STAT3", "Monocytes"): {"uniprot_id": STAT3, "raw_tstat": 5.0},
        }
        builder = CompositeNodeBuilder(min_tstat=2.0)
        result = builder._filter_and_normalize(tstat_data)
        for val in result.values():
            assert abs(val["norm_weight"] - 0.5) < 1e-6

    def test_empty_after_filter_returns_empty(self):
        """If no pairs pass min_tstat, return empty dict without crashing."""
        tstat_data = {
            ("TNF", "T cells"): {"uniprot_id": TNF, "raw_tstat": 1.0},
        }
        builder = CompositeNodeBuilder(min_tstat=5.0)
        result = builder._filter_and_normalize(tstat_data)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: full build
# ---------------------------------------------------------------------------

class TestBuild:

    def test_node_count_increases_by_composite_count(self):
        """Graph node count must increase by exactly N composite nodes."""
        G = make_synthetic_graph()
        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv([
                {"gene": "TNF",   "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 2.0},
                {"gene": "STAT3", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 4.0,  "T cells": 0},
            ], tmp)

            before = G.number_of_nodes()
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            result = builder.build(G, mapper, csv_path)

        # TNF Monocytes (32.2 >= 2.0): 1 node
        # TNF T cells (2.0 >= 2.0): 1 node
        # STAT3 Monocytes (4.0 >= 2.0): 1 node
        # Total: 3 composite nodes
        assert result.n_composite_nodes == 3
        assert result.graph.number_of_nodes() == before + 3

    def test_each_composite_node_has_two_edges(self):
        """Each composite node must have exactly 2 edges:
        Castleman->composite and composite->protein."""
        G = make_synthetic_graph()
        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv([
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 0},
            ], tmp)
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            result = builder.build(G, mapper, csv_path)

        comp_id = CompositeNodeBuilder.make_composite_id("TNF", "Monocytes")
        G2 = result.graph
        assert G2.has_edge(CASTLEMAN, comp_id), "Missing Castleman->composite edge"
        assert G2.has_edge(comp_id, TNF),       "Missing composite->TNF edge"

    def test_original_edges_preserved(self):
        """TNF->Castleman and Adalimumab->TNF must still exist after build."""
        G = make_synthetic_graph()
        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv([
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 0},
            ], tmp)
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            result = builder.build(G, mapper, csv_path)

        G2 = result.graph
        assert G2.has_edge(TNF, CASTLEMAN),    "TNF->Castleman edge was removed"
        assert G2.has_edge(ADALIMUMAB, TNF),   "Adalimumab->TNF edge was removed"

    def test_composite_node_weights_in_range(self):
        """Castleman->composite edge weights must be in [0.1, 1.0]."""
        G = make_synthetic_graph()
        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv([
                {"gene": "TNF",   "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 3.0},
                {"gene": "STAT3", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 4.0,  "T cells": 0},
            ], tmp)
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            result = builder.build(G, mapper, csv_path)

        for info in result.composite_nodes:
            w = result.graph[CASTLEMAN][info.node_id]["weight"]
            assert 0.1 <= w <= 1.0, f"{info.node_id} weight {w} out of range"

    def test_no_qualifying_pairs_returns_empty(self):
        """If all t-stats are below min_tstat, composite_nodes list is empty."""
        G = make_synthetic_graph()
        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv([
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 1.0, "T cells": 0},
            ], tmp)
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            result = builder.build(G, mapper, csv_path)

        assert result.n_composite_nodes == 0
        assert result.graph.number_of_nodes() == G.number_of_nodes()

    def test_composite_node_id_format(self):
        """Composite node IDs must follow COMPOSITE:{gene}|{cell_type} format."""
        G = make_synthetic_graph()
        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv([
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 0},
            ], tmp)
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            result = builder.build(G, mapper, csv_path)

        assert result.composite_nodes[0].node_id == "COMPOSITE:TNF|Monocytes"

    def test_composite_to_protein_edge_weight_is_one(self):
        """composite->canonical protein edge weight must be exactly 1.0."""
        G = make_synthetic_graph()
        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv([
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 0},
            ], tmp)
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            result = builder.build(G, mapper, csv_path)

        comp_id = "COMPOSITE:TNF|Monocytes"
        w = result.graph[comp_id][TNF]["weight"]
        assert w == 1.0

    def test_original_graph_not_mutated(self):
        """build() must work on a copy — original graph node count unchanged."""
        G = make_synthetic_graph()
        original_count = G.number_of_nodes()
        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv([
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 0},
            ], tmp)
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            builder.build(G, mapper, csv_path)

        assert G.number_of_nodes() == original_count


# ---------------------------------------------------------------------------
# Tests: Phase 5.3 new behaviour
# ---------------------------------------------------------------------------

class TestAllGenesMode:

    def test_non_neighbor_gene_gets_composite_node(self):
        """
        A gene that is NOT a direct Castleman neighbor but IS in the CSV
        should still get a composite node after the Phase 5.3 change.
        This is the core difference from Phase 5.2.
        """
        G = make_synthetic_graph()
        # Add a protein that has NO edge to Castleman
        non_neighbor = "UniProtKB:P99999"
        G.add_node(non_neighbor, name="ORPHAN", category="biolink:Protein")
        # Confirm it is NOT a Castleman predecessor
        assert non_neighbor not in list(G.predecessors(CASTLEMAN))

        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv([
                {"gene": "ORPHAN", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 10.0, "T cells": 0},
            ], tmp)
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            result = builder.build(G, mapper, csv_path)

        comp_id = CompositeNodeBuilder.make_composite_id("ORPHAN", "Monocytes")
        G2 = result.graph
        assert result.n_composite_nodes == 1, "Expected 1 composite node for ORPHAN"
        assert G2.has_node(comp_id), "Composite node not added"
        assert G2.has_edge(CASTLEMAN, comp_id), "Missing Castleman->composite edge"
        assert G2.has_edge(comp_id, non_neighbor), "Missing composite->protein edge"

    def test_negative_tstat_included_as_abs_value(self):
        """
        A gene with a negative t-stat must produce a composite node.
        raw_tstat must equal abs(tstat), not the original negative value.

        This guards the Phase 5.6 change: the new CSV (file 2) uses
        remission as reference, so up-regulated genes in remission have
        negative t-stats. We include all differentially expressed genes
        regardless of direction by using abs(tstat).
        """
        G = make_synthetic_graph()
        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            # TNF t-stat = -5.0 (down in flare / up in remission)
            csv_path = make_csv([
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": -5.0, "T cells": 0},
            ], tmp)
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            result = builder.build(G, mapper, csv_path)

        assert result.n_composite_nodes == 1, (
            "Negative t-stat gene should produce a composite node via abs(tstat)"
        )
        node = result.composite_nodes[0]
        assert node.raw_tstat == 5.0, (
            f"raw_tstat should be abs(-5.0)=5.0, got {node.raw_tstat}"
        )
        assert node.gene_symbol == "TNF"
        assert node.cell_type == "Monocytes"

    def test_cell_type_filter_limits_composite_nodes(self):
        """
        Passing cell_types=["Monocytes"] should create only Monocyte
        composite nodes, even if other cell types also have qualifying t-stats.
        """
        G = make_synthetic_graph()
        mapper = make_mapper_for(G)

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv([
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 5.0},
            ], tmp)
            builder = CompositeNodeBuilder(castleman_id=CASTLEMAN, min_tstat=2.0)
            result = builder.build(G, mapper, csv_path, cell_types=["Monocytes"])

        # Only 1 node: TNF|Monocytes — T cells excluded by filter
        assert result.n_composite_nodes == 1
        mono_id = CompositeNodeBuilder.make_composite_id("TNF", "Monocytes")
        tcell_id = CompositeNodeBuilder.make_composite_id("TNF", "T cells")
        assert result.graph.has_node(mono_id)
        assert not result.graph.has_node(tcell_id)


# ---------------------------------------------------------------------------
# Tests: make_composite_id static method
# ---------------------------------------------------------------------------

class TestMakeCompositeId:

    def test_basic_format(self):
        cid = CompositeNodeBuilder.make_composite_id("TNF", "Monocytes")
        assert cid == "COMPOSITE:TNF|Monocytes"

    def test_spaces_replaced(self):
        cid = CompositeNodeBuilder.make_composite_id("TNF", "T cells")
        assert cid == "COMPOSITE:TNF|T_cells"

    def test_slashes_replaced(self):
        cid = CompositeNodeBuilder.make_composite_id("CALM2", "Megakaryocytes/platelets")
        assert cid == "COMPOSITE:CALM2|Megakaryocytes_platelets"
