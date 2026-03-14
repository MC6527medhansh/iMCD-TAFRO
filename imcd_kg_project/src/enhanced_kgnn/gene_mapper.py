"""
gene_mapper.py
--------------
Purpose: Map gene symbols (e.g., "TNF") from the iMCD-TAFRO scRNA-seq CSV
         to UniProtKB node IDs (e.g., "UniProtKB:P01375") that exist in
         RTX-KG2 (our graph).

Why this is needed:
  The CSV uses human-readable gene symbols. The graph uses UniProtKB CURIEs.
  Before building any composite nodes or edge weights, we need to know:
    (a) How many of the 12,500 CSV genes exist in our graph?
    (b) What are their UniProtKB IDs so we can reference them later?

How it works:
  1. Load the NetworkX graph from full_graph.pkl
  2. For every node with a "UniProtKB:" prefix, read its 'name' attribute
     (RTX-KG2 stores the gene symbol in the node's 'name' field)
  3. Build a reverse dict: gene_symbol (uppercase) -> UniProtKB ID
  4. For each gene in the CSV, look it up in the dict
  5. Return matched, unmatched, and a coverage percentage

Usage:
  mapper = GeneMapper(graph_path)
  mapper.load_graph()
  report = mapper.map_csv(csv_path)
  print(report.summary())
"""

import csv
import pickle
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class MappingReport:
    """
    Results of mapping CSV gene symbols to graph UniProtKB IDs.

    Attributes:
        matched:    Dict of gene_symbol -> UniProtKB ID for successful matches
        unmatched:  List of gene symbols with no node in the graph
        csv_total:  Total number of genes in the CSV
        cell_types: Cell type column names from the CSV
    """
    matched: Dict[str, str] = field(default_factory=dict)
    unmatched: List[str] = field(default_factory=list)
    csv_total: int = 0
    cell_types: List[str] = field(default_factory=list)

    @property
    def coverage_pct(self) -> float:
        """Percentage of CSV genes that have a matching node in the graph."""
        if self.csv_total == 0:
            return 0.0
        return 100.0 * len(self.matched) / self.csv_total

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Gene Symbol -> UniProtKB Mapping Report",
            "=" * 60,
            f"CSV total genes:    {self.csv_total:,}",
            f"Matched to graph:   {len(self.matched):,}",
            f"Unmatched:          {len(self.unmatched):,}",
            f"Coverage:           {self.coverage_pct:.1f}%",
            f"Cell types:         {self.cell_types}",
            "=" * 60,
        ]
        return "\n".join(lines)


class GeneMapper:
    """
    Maps gene symbols from the scRNA-seq CSV to UniProtKB IDs in the graph.

    The graph stores protein nodes with IDs like "UniProtKB:P01375" and a
    'name' attribute containing the gene symbol (e.g., "TNF"). This class
    builds the reverse lookup and reports coverage against the CSV.

    Args:
        graph_path: Path to full_graph.pkl (on Sockeye) or a synthetic
                    NetworkX graph (for local testing).
    """

    # Cell type columns in the CSV (used to skip non-gene columns)
    EXPECTED_CELL_TYPES = [
        "B cells",
        "ILC",
        "Megakaryocytes/platelets",
        "Monocytes",
        "T cells",
    ]

    def __init__(self, graph_path: Optional[Path] = None):
        self.graph_path = graph_path
        self._nx_graph = None
        # symbol (uppercase) -> UniProtKB ID
        self._symbol_to_id: Dict[str, str] = {}

    def load_graph(self, nx_graph=None) -> None:
        """
        Load the graph and build the gene symbol -> UniProtKB ID lookup.

        Args:
            nx_graph: Pre-loaded NetworkX graph (for testing without disk I/O).
                      If None, loads from self.graph_path.

        Raises:
            FileNotFoundError: If graph_path does not exist.
            ValueError: If graph has no nodes with UniProtKB prefix.
        """
        if nx_graph is not None:
            self._nx_graph = nx_graph
            logger.info("Using provided NetworkX graph (test mode)")
        else:
            if self.graph_path is None:
                raise ValueError("graph_path must be provided if nx_graph is None")
            if not Path(self.graph_path).exists():
                raise FileNotFoundError(
                    f"Graph not found: {self.graph_path}\n"
                    "Run Phase 1.2 on Sockeye to generate full_graph.pkl"
                )
            with open(self.graph_path, "rb") as f:
                self._nx_graph = pickle.load(f)
            logger.info(
                f"Loaded graph: {self._nx_graph.number_of_nodes():,} nodes, "
                f"{self._nx_graph.number_of_edges():,} edges"
            )

        self._symbol_to_id = self._build_symbol_lookup()
        logger.info(
            f"Built symbol lookup: {len(self._symbol_to_id):,} "
            "UniProtKB nodes with gene symbols"
        )

    def _build_symbol_lookup(self) -> Dict[str, str]:
        """
        Walk every node in the graph. For UniProtKB nodes, extract the
        gene symbol from the node's 'name' attribute and store it.

        RTX-KG2 node attributes (example for TNF):
          node_id:  "UniProtKB:P01375"
          data:     {'name': 'TNF', 'category': 'biolink:Protein', ...}

        Returns:
            Dict mapping UPPERCASE gene symbol -> UniProtKB CURIE ID.
            Upper-cased to make matching case-insensitive.
        """
        symbol_to_id: Dict[str, str] = {}

        for node_id, data in self._nx_graph.nodes(data=True):
            if not node_id.startswith("UniProtKB:"):
                continue

            # Try 'name' attribute first (RTX-KG2 standard)
            name = data.get("name") or data.get("symbol") or data.get("label")
            if not name:
                continue

            symbol = str(name).strip().upper()
            if symbol:
                # Keep first match if duplicates exist (shouldn't happen in RTX-KG2)
                if symbol not in symbol_to_id:
                    symbol_to_id[symbol] = node_id

        return symbol_to_id

    def get_uniprot_id(self, gene_symbol: str) -> Optional[str]:
        """
        Look up the UniProtKB ID for a given gene symbol.

        Args:
            gene_symbol: Human gene symbol, e.g. "TNF" or "tnf" (case-insensitive)

        Returns:
            UniProtKB CURIE string (e.g. "UniProtKB:P01375") or None if not found.
        """
        if not self._symbol_to_id:
            raise RuntimeError("Call load_graph() before get_uniprot_id()")
        return self._symbol_to_id.get(gene_symbol.strip().upper())

    def map_csv(self, csv_path: Path) -> MappingReport:
        """
        Map all gene symbols in the CSV to UniProtKB IDs.

        Reads the gene column from the CSV, looks up each symbol, and
        returns a MappingReport with matched/unmatched genes.

        Args:
            csv_path: Path to iMCD_TAFRO_cell_specific_tstats.csv

        Returns:
            MappingReport with matched dict, unmatched list, coverage %.

        Raises:
            RuntimeError: If load_graph() has not been called.
            FileNotFoundError: If csv_path does not exist.
        """
        if not self._symbol_to_id:
            raise RuntimeError("Call load_graph() before map_csv()")

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        report = MappingReport()

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            # Cell type columns = everything except the index column and 'gene'
            report.cell_types = [
                h for h in headers if h not in ("", "gene")
            ]

            for row in reader:
                gene_symbol = row.get("gene", "").strip()
                if not gene_symbol:
                    continue

                report.csv_total += 1
                uniprot_id = self.get_uniprot_id(gene_symbol)

                if uniprot_id is not None:
                    report.matched[gene_symbol] = uniprot_id
                else:
                    report.unmatched.append(gene_symbol)

        logger.info(report.summary())
        return report

    def map_csv_with_tstats(
        self,
        csv_path: Path,
        min_tstat: float = 0.0,
    ) -> Dict[str, Dict[str, float]]:
        """
        Map CSV to graph IDs AND return t-stats per matched gene.

        Returns a nested dict:
          {
            gene_symbol: {
              cell_type: t_stat_value,
              ...
            },
            ...
          }
        Only returns genes that (a) exist in the graph and
        (b) have at least one non-zero t-stat across cell types.

        Args:
            csv_path:  Path to the CSV file.
            min_tstat: Only include (gene, cell_type) pairs where
                       t-stat >= min_tstat. Default 0.0 keeps all non-zero.
        """
        if not self._symbol_to_id:
            raise RuntimeError("Call load_graph() before map_csv_with_tstats()")

        result: Dict[str, Dict[str, float]] = {}

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            cell_types = [h for h in headers if h not in ("", "gene")]

            for row in reader:
                gene_symbol = row.get("gene", "").strip()
                if not gene_symbol:
                    continue

                uniprot_id = self.get_uniprot_id(gene_symbol)
                if uniprot_id is None:
                    continue

                cell_tstats: Dict[str, float] = {}
                for ct in cell_types:
                    try:
                        val = float(row[ct])
                    except (ValueError, KeyError):
                        val = 0.0
                    if val >= min_tstat:
                        cell_tstats[ct] = val

                if any(v > 0 for v in cell_tstats.values()):
                    result[gene_symbol] = {
                        "uniprot_id": uniprot_id,
                        **cell_tstats,
                    }

        logger.info(
            f"map_csv_with_tstats: {len(result):,} matched genes with "
            f"non-zero t-stats (min_tstat={min_tstat})"
        )
        return result
