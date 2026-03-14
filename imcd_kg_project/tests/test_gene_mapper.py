"""
test_gene_mapper.py
-------------------
Unit tests for gene_mapper.py

All tests run locally using a synthetic NetworkX graph — no full_graph.pkl
needed. This lets us test the logic without Sockeye access.

What is being tested:
  1. TNF maps to the correct UniProtKB ID (ground truth we know)
  2. Non-UniProtKB nodes (drugs, diseases) are not included in the lookup
  3. Case-insensitive matching works ("tnf" == "TNF")
  4. Missing genes return None without crashing
  5. Coverage percentage is computed correctly
  6. map_csv correctly parses the real CSV format
  7. map_csv_with_tstats returns correct t-stat values
  8. load_graph raises FileNotFoundError for missing graph file
  9. Calling map_csv before load_graph raises RuntimeError
"""

import csv
import os
import tempfile
import sys
from pathlib import Path

import networkx as nx
import pytest

# Add src to path so we can import gene_mapper
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from enhanced_kgnn.gene_mapper import GeneMapper, MappingReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_synthetic_graph() -> nx.Graph:
    """
    Build a small fake graph that mirrors RTX-KG2 node structure.

    Node structure:
      - UniProtKB nodes have a 'name' attribute (gene symbol)
      - MONDO nodes (diseases) have no 'name' that should be picked up
      - CHEMBL nodes (drugs) have no 'name' matching gene symbols

    Included proteins:
      UniProtKB:P01375 = TNF
      UniProtKB:P05231 = IL6
      UniProtKB:P42224 = STAT3
      UniProtKB:P22301 = IL10
      UniProtKB:P10145 = IL8  (CXCL8 in newer nomenclature)
    """
    G = nx.Graph()

    # Protein nodes (UniProtKB prefix, 'name' = gene symbol)
    G.add_node("UniProtKB:P01375", name="TNF",   category="biolink:Protein")
    G.add_node("UniProtKB:P05231", name="IL6",   category="biolink:Protein")
    G.add_node("UniProtKB:P42224", name="STAT3", category="biolink:Protein")
    G.add_node("UniProtKB:P22301", name="IL10",  category="biolink:Protein")
    G.add_node("UniProtKB:P10145", name="IL8",   category="biolink:Protein")

    # Disease nodes (should NOT appear in symbol lookup)
    G.add_node("MONDO:0015564", name="Castleman Disease", category="biolink:Disease")
    G.add_node("MONDO:0005148", name="Type 2 Diabetes",   category="biolink:Disease")

    # Drug nodes (should NOT appear in symbol lookup)
    G.add_node("CHEMBL.COMPOUND:CHEMBL1201580", name="adalimumab", category="biolink:Drug")

    # Some edges (content doesn't matter for mapper tests)
    G.add_edge("MONDO:0015564", "UniProtKB:P01375")
    G.add_edge("CHEMBL.COMPOUND:CHEMBL1201580", "UniProtKB:P01375")

    return G


def make_csv_file(rows: list, tmp_dir: str) -> Path:
    """
    Write a minimal CSV file mimicking iMCD_TAFRO_cell_specific_tstats.csv format.

    Args:
        rows: list of dicts with keys: gene, B cells, ILC,
              Megakaryocytes/platelets, Monocytes, T cells
        tmp_dir: directory to write the file into

    Returns:
        Path to the written CSV file.
    """
    path = Path(tmp_dir) / "test_tstats.csv"
    fieldnames = ["", "gene", "B cells", "ILC", "Megakaryocytes/platelets",
                  "Monocytes", "T cells"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow({"": str(i), **row})
    return path


# ---------------------------------------------------------------------------
# Tests: load_graph
# ---------------------------------------------------------------------------

class TestLoadGraph:

    def test_load_graph_from_synthetic(self):
        """load_graph accepts a pre-built nx graph (no disk I/O)."""
        g = make_synthetic_graph()
        mapper = GeneMapper()
        mapper.load_graph(nx_graph=g)
        # Should build a lookup with 5 proteins
        assert len(mapper._symbol_to_id) == 5

    def test_load_graph_missing_file_raises(self):
        """load_graph raises FileNotFoundError if graph_path does not exist."""
        mapper = GeneMapper(graph_path=Path("/nonexistent/full_graph.pkl"))
        with pytest.raises(FileNotFoundError):
            mapper.load_graph()

    def test_load_graph_no_path_no_graph_raises(self):
        """load_graph raises ValueError if neither path nor graph is given."""
        mapper = GeneMapper()
        with pytest.raises(ValueError):
            mapper.load_graph()


# ---------------------------------------------------------------------------
# Tests: get_uniprot_id
# ---------------------------------------------------------------------------

class TestGetUniprotId:

    @pytest.fixture(autouse=True)
    def setup(self):
        g = make_synthetic_graph()
        self.mapper = GeneMapper()
        self.mapper.load_graph(nx_graph=g)

    def test_tnf_maps_to_correct_id(self):
        """TNF maps to UniProtKB:P01375 — our known ground truth."""
        result = self.mapper.get_uniprot_id("TNF")
        assert result == "UniProtKB:P01375", (
            f"Expected UniProtKB:P01375, got {result}"
        )

    def test_il6_maps_correctly(self):
        result = self.mapper.get_uniprot_id("IL6")
        assert result == "UniProtKB:P05231"

    def test_case_insensitive_lowercase(self):
        """Gene symbols should match regardless of case."""
        assert self.mapper.get_uniprot_id("tnf") == "UniProtKB:P01375"

    def test_case_insensitive_mixed(self):
        assert self.mapper.get_uniprot_id("Il6") == "UniProtKB:P05231"

    def test_missing_gene_returns_none(self):
        """A gene symbol not in the graph returns None, not an exception."""
        result = self.mapper.get_uniprot_id("FAKEGENE999")
        assert result is None

    def test_disease_nodes_not_in_lookup(self):
        """MONDO disease nodes should never appear in the symbol lookup."""
        # "Castleman Disease" is a node name but is MONDO, not UniProtKB
        result = self.mapper.get_uniprot_id("Castleman Disease")
        assert result is None

    def test_drug_nodes_not_in_lookup(self):
        """CHEMBL drug nodes should not appear in the symbol lookup."""
        result = self.mapper.get_uniprot_id("adalimumab")
        assert result is None

    def test_get_uniprot_before_load_raises(self):
        """get_uniprot_id raises RuntimeError if called before load_graph."""
        fresh_mapper = GeneMapper()
        with pytest.raises(RuntimeError):
            fresh_mapper.get_uniprot_id("TNF")


# ---------------------------------------------------------------------------
# Tests: map_csv
# ---------------------------------------------------------------------------

class TestMapCsv:

    @pytest.fixture(autouse=True)
    def setup(self):
        g = make_synthetic_graph()
        self.mapper = GeneMapper()
        self.mapper.load_graph(nx_graph=g)

    def test_map_csv_basic_coverage(self):
        """CSV with 3 genes: 2 in graph, 1 not -> coverage 66.7%."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv_file([
                {"gene": "TNF",      "B cells": 1.0, "ILC": 0,   "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 2.9},
                {"gene": "IL6",      "B cells": 0.5, "ILC": 1.5, "Megakaryocytes/platelets": 0, "Monocytes": 0,    "T cells": 1.0},
                {"gene": "FAKEGENE", "B cells": 0,   "ILC": 0,   "Megakaryocytes/platelets": 0, "Monocytes": 0,    "T cells": 0},
            ], tmp)
            report = self.mapper.map_csv(csv_path)

        assert report.csv_total == 3
        assert len(report.matched) == 2
        assert len(report.unmatched) == 1
        assert "FAKEGENE" in report.unmatched
        assert abs(report.coverage_pct - 66.67) < 0.1

    def test_map_csv_tnf_matched(self):
        """TNF gene in CSV maps to UniProtKB:P01375."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv_file([
                {"gene": "TNF", "B cells": 0, "ILC": 0, "Megakaryocytes/platelets": 0,
                 "Monocytes": 32.2, "T cells": 2.9},
            ], tmp)
            report = self.mapper.map_csv(csv_path)

        assert "TNF" in report.matched
        assert report.matched["TNF"] == "UniProtKB:P01375"

    def test_map_csv_cell_types_detected(self):
        """map_csv correctly identifies cell type column names."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv_file([
                {"gene": "TNF", "B cells": 0, "ILC": 0, "Megakaryocytes/platelets": 0,
                 "Monocytes": 32.2, "T cells": 2.9},
            ], tmp)
            report = self.mapper.map_csv(csv_path)

        assert "B cells" in report.cell_types
        assert "T cells" in report.cell_types
        assert "Monocytes" in report.cell_types
        assert "gene" not in report.cell_types

    def test_map_csv_before_load_raises(self):
        """map_csv raises RuntimeError if called before load_graph."""
        fresh_mapper = GeneMapper()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv_file([
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 0, "T cells": 0}
            ], tmp)
            with pytest.raises(RuntimeError):
                fresh_mapper.map_csv(csv_path)

    def test_map_csv_missing_file_raises(self):
        """map_csv raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.mapper.map_csv(Path("/nonexistent/tstats.csv"))

    def test_coverage_pct_zero_when_no_genes(self):
        """coverage_pct returns 0.0 when csv_total is 0 (no divide by zero)."""
        report = MappingReport()
        assert report.coverage_pct == 0.0


# ---------------------------------------------------------------------------
# Tests: map_csv_with_tstats
# ---------------------------------------------------------------------------

class TestMapCsvWithTstats:

    @pytest.fixture(autouse=True)
    def setup(self):
        g = make_synthetic_graph()
        self.mapper = GeneMapper()
        self.mapper.load_graph(nx_graph=g)

    def test_returns_tstats_for_matched_gene(self):
        """Matched genes have their t-stat values accessible per cell type."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv_file([
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 2.9},
            ], tmp)
            result = self.mapper.map_csv_with_tstats(csv_path)

        assert "TNF" in result
        assert result["TNF"]["uniprot_id"] == "UniProtKB:P01375"
        assert abs(result["TNF"]["Monocytes"] - 32.2) < 0.01
        assert abs(result["TNF"]["T cells"] - 2.9) < 0.01

    def test_unmatched_gene_excluded(self):
        """Genes not in the graph are excluded from the result."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv_file([
                {"gene": "FAKEGENE", "B cells": 5.0, "ILC": 3.0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 1.0, "T cells": 2.0},
            ], tmp)
            result = self.mapper.map_csv_with_tstats(csv_path)

        assert "FAKEGENE" not in result

    def test_min_tstat_filter(self):
        """min_tstat=5.0 keeps only (gene, cell_type) pairs with t-stat >= 5."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv_file([
                # TNF: Monocytes=32.2 (keep), T cells=2.9 (filter out)
                {"gene": "TNF", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 2.9},
            ], tmp)
            result = self.mapper.map_csv_with_tstats(csv_path, min_tstat=5.0)

        assert "TNF" in result
        assert result["TNF"]["Monocytes"] == 32.2
        # T cells t-stat (2.9) is below 5.0 — should NOT be included
        assert "T cells" not in result["TNF"]

    def test_all_zero_gene_excluded(self):
        """A gene with all-zero t-stats is excluded even if it maps to the graph."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = make_csv_file([
                {"gene": "IL6", "B cells": 0, "ILC": 0,
                 "Megakaryocytes/platelets": 0, "Monocytes": 0, "T cells": 0},
            ], tmp)
            result = self.mapper.map_csv_with_tstats(csv_path)

        assert "IL6" not in result
