"""
composite_node_builder.py
-------------------------
Purpose: Insert cell-type-specific composite nodes into the RTX-KG2 graph.

What this does (plain English):
  For every gene in the scRNA-seq CSV that maps to a protein node in the
  graph, this module creates one composite intermediate node per cell type
  with a t-stat >= min_tstat.

  Before:
    Castleman  <-  TNF  <-  Adalimumab

  After (with composite nodes):
    Castleman  <-  [TNF|Monocytes]  <-  TNF  <-  Adalimumab
    Castleman  <-  [TNF|T_cells]    <-  TNF

  Note: the original TNF -> Castleman edge is kept. The composite nodes
  add NEW paths, they do not replace anything.

  Previously (Phase 5.2) we only created composite nodes for genes that
  were ALREADY direct neighbors of Castleman in the graph. That gave 49
  composite nodes.

  Now we use ALL ~10,900 genes in the CSV that map to the graph. This
  gives Castleman connections to thousands of cell-type-specific proteins
  instead of just 49. The t-stat IS the evidence of relevance — we do not
  need the graph to have pre-existing Castleman edges for those genes.

  Composite node IDs look like: "COMPOSITE:TNF|Monocytes"
  These nodes get node feature [0, 0, 1] (same as other proteins).

Cell-type filtering:
  Pass cell_types=["T cells"] to build() to create composite nodes for
  only one cell type. This is used for per-cell-type experiments.
  Default (cell_types=None) uses all five cell types.

Normalization strategy:
  Raw t-stats range from 0 to ~96. Feeding these directly into GATConv
  attention would dominate the softmax. We clip at the 99th percentile
  then min-max scale to [0.1, 1.0]. The 0.1 floor ensures no edge weight
  is zero (zero weight = invisible edge).
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompositeNodeInfo:
    """Metadata for one composite node."""
    node_id: str        # e.g. "COMPOSITE:TNF|Monocytes"
    gene_symbol: str    # e.g. "TNF"
    cell_type: str      # e.g. "Monocytes"
    uniprot_id: str     # e.g. "UniProtKB:P01375"
    raw_tstat: float    # original t-stat from CSV
    norm_weight: float  # normalized weight in [0.1, 1.0]


@dataclass
class BuildResult:
    """Output of CompositeNodeBuilder.build()."""
    graph: nx.DiGraph                          # modified graph with composite nodes
    composite_nodes: List[CompositeNodeInfo]   # metadata for each composite node
    n_original_nodes: int                      # node count before modification
    n_original_edges: int                      # edge count before modification

    @property
    def n_composite_nodes(self) -> int:
        return len(self.composite_nodes)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Composite Node Build Summary",
            "=" * 60,
            f"Original nodes:    {self.n_original_nodes:,}",
            f"Original edges:    {self.n_original_edges:,}",
            f"Composite nodes added: {self.n_composite_nodes:,}",
            f"New total nodes:   {self.graph.number_of_nodes():,}",
            f"New total edges:   {self.graph.number_of_edges():,}",
            "=" * 60,
        ]
        return "\n".join(lines)


class CompositeNodeBuilder:
    """
    Inserts composite (gene, cell_type) intermediate nodes into the graph.

    Usage:
        builder = CompositeNodeBuilder(
            castleman_id="MONDO:0015564",
            min_tstat=2.0,
        )
        result = builder.build(nx_graph, gene_mapper, csv_path)
        # result.graph is the modified graph
        # result.composite_nodes has metadata for each new node
    """

    def __init__(
        self,
        castleman_id: str = "MONDO:0015564",
        min_tstat: float = 2.0,
    ):
        self.castleman_id = castleman_id
        self.min_tstat = min_tstat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        nx_graph: nx.DiGraph,
        gene_mapper,
        csv_path: Path,
        cell_types: Optional[List[str]] = None,
    ) -> BuildResult:
        """
        Build the composite-node-augmented graph.

        Steps:
          1. Load t-stats for ALL genes in the CSV that map to the graph.
             (Phase 5.2 only used Castleman's 68 direct neighbors.
              Now we use all ~10,900 matched genes — the t-stat IS the
              evidence of relevance, we do not need a pre-existing edge.)
          2. Optionally restrict to specific cell types (for per-cell-type
             experiments). Default None uses all five cell types.
          3. Filter by min_tstat.
          4. Normalize t-stats to [0.1, 1.0].
          5. For each (gene, cell_type) pair, insert a composite node
             with two new edges:
               Castleman -> composite (weight = norm_tstat)
               composite -> canonical_protein (weight = 1.0)
          6. Return modified graph + metadata.

        Args:
            nx_graph:    Directed NetworkX graph (RTX-KG2 full_graph)
            gene_mapper: Loaded GeneMapper instance
            csv_path:    Path to iMCD_TAFRO_cell_specific_tstats.csv
            cell_types:  Optional list of cell type column names to include.
                         Example: ["T cells"] for a T-cells-only experiment.
                         None (default) uses all cell types in the CSV.

        Returns:
            BuildResult with modified graph and composite node metadata.
        """
        n_orig_nodes = nx_graph.number_of_nodes()
        n_orig_edges = nx_graph.number_of_edges()

        # Work on a copy — never mutate the original graph
        G = nx_graph.copy()

        # Log direct Castleman neighbors for reference (informational only)
        protein_neighbors = self._get_protein_neighbors(G)
        logger.info(
            f"Castleman has {len(protein_neighbors)} incoming protein neighbors "
            f"(pre-existing graph edges)"
        )

        # Step 1+2: Load t-stats for ALL matched genes, filter cell types
        tstat_data = self._load_tstats_all_genes(csv_path, gene_mapper, cell_types)
        logger.info(
            f"Loaded {len(tstat_data)} (gene, cell_type) pairs from CSV "
            f"(cell_types filter: {cell_types})"
        )

        # Step 3 + 4: Filter by min_tstat and normalize
        filtered = self._filter_and_normalize(tstat_data)
        logger.info(
            f"After min_tstat={self.min_tstat} filter: "
            f"{len(filtered)} (gene, cell_type) pairs"
        )

        # Step 5: Insert composite nodes into graph
        composite_nodes = self._insert_composite_nodes(G, filtered)

        return BuildResult(
            graph=G,
            composite_nodes=composite_nodes,
            n_original_nodes=n_orig_nodes,
            n_original_edges=n_orig_edges,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_protein_neighbors(self, G: nx.DiGraph) -> Dict[str, str]:
        """
        Get Castleman's incoming protein neighbors.

        The graph is directed. Proteins point TO Castleman
        (protein -> Castleman). We use G.predecessors() to get them.

        Returns:
            Dict mapping UniProtKB ID -> gene symbol (from node 'name' attr)
        """
        neighbors = {}
        for node_id in G.predecessors(self.castleman_id):
            if not node_id.startswith("UniProtKB:"):
                continue
            gene_name = G.nodes[node_id].get("name", "")
            if gene_name:
                neighbors[node_id] = gene_name
        return neighbors

    def _load_tstats_all_genes(
        self,
        csv_path: Path,
        gene_mapper,
        cell_types_filter: Optional[List[str]] = None,
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Load t-stats for ALL genes in the CSV that map to the graph.

        Unlike the old _load_tstats_for_neighbors, this does not require
        the gene to already have an edge to Castleman in the graph. Any gene
        that maps to a UniProtKB ID is included. The t-stat is the evidence
        of biological relevance — not the pre-existing graph topology.

        Args:
            csv_path:           Path to the t-stat CSV.
            gene_mapper:        Loaded GeneMapper instance.
            cell_types_filter:  If provided, only load these cell type columns.
                                If None, load all cell type columns in the CSV.

        Returns:
            Dict mapping (gene_symbol, cell_type) -> {
                'uniprot_id': str,
                'raw_tstat': float,
            }
            Only includes pairs where abs(tstat) > 0.
            raw_tstat stores abs(tstat) — direction (up/down in flare) is
            ignored because any differential expression is evidence of
            cell-type involvement regardless of sign.
        """
        result: Dict[Tuple[str, str], Dict] = {}

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            cell_types = [h for h in headers if h not in ("", "gene")]

            if cell_types_filter is not None:
                cell_types = [ct for ct in cell_types if ct in cell_types_filter]
                if not cell_types:
                    logger.warning(
                        f"cell_types_filter {cell_types_filter} matched none of "
                        f"the CSV columns {headers}. No composite nodes will be "
                        f"created. Check for typos in cell type names."
                    )

            for row in reader:
                gene_symbol = row.get("gene", "").strip()
                if not gene_symbol:
                    continue

                uniprot_id = gene_mapper.get_uniprot_id(gene_symbol)
                if uniprot_id is None:
                    continue

                for ct in cell_types:
                    try:
                        tstat = float(row.get(ct, 0))
                    except ValueError:
                        tstat = 0.0

                    abs_tstat = abs(tstat)
                    if abs_tstat > 0:
                        result[(gene_symbol, ct)] = {
                            "uniprot_id": uniprot_id,
                            "raw_tstat": abs_tstat,
                        }

        return result

    def _filter_and_normalize(
        self,
        tstat_data: Dict[Tuple[str, str], Dict],
    ) -> Dict[Tuple[str, str], Dict]:
        """
        Filter by min_tstat, then normalize raw t-stats to [0.1, 1.0].

        Normalization steps:
          1. Keep only pairs where raw_tstat >= min_tstat
          2. Clip values at the 99th percentile (removes extreme outliers)
          3. Min-max scale to [0.1, 1.0]
             - 0.1 floor ensures no edge weight is exactly zero
             - Preserves relative ordering of t-stats

        Returns:
            Same structure as input but with 'norm_weight' added and
            only pairs passing the min_tstat filter.
        """
        # Step 1: filter
        filtered = {
            k: v for k, v in tstat_data.items()
            if v["raw_tstat"] >= self.min_tstat
        }

        if not filtered:
            logger.warning(
                f"No (gene, cell_type) pairs pass min_tstat={self.min_tstat}"
            )
            return filtered

        # Step 2: clip at 99th percentile
        raw_values = np.array([v["raw_tstat"] for v in filtered.values()])
        p99 = float(np.percentile(raw_values, 99))
        clipped = np.clip(raw_values, a_min=None, a_max=p99)

        # Step 3: min-max scale to [0.1, 1.0]
        c_min = float(clipped.min())
        c_max = float(clipped.max())

        if c_max == c_min:
            # All values identical — assign uniform weight 0.5
            norm_values = np.full_like(clipped, 0.5)
        else:
            norm_values = 0.1 + 0.9 * (clipped - c_min) / (c_max - c_min)

        for (key, val), norm_w in zip(filtered.items(), norm_values):
            val["norm_weight"] = float(norm_w)

        logger.info(
            f"Normalization: 99th pct clip={p99:.2f}, "
            f"weight range=[{norm_values.min():.3f}, {norm_values.max():.3f}]"
        )
        return filtered

    def _insert_composite_nodes(
        self,
        G: nx.DiGraph,
        filtered: Dict[Tuple[str, str], Dict],
    ) -> List[CompositeNodeInfo]:
        """
        Insert composite nodes and their edges into the graph.

        For each (gene, cell_type) pair:
          - Create node: "COMPOSITE:{gene}|{cell_type}"
          - Add edge: Castleman -> composite  (weight = norm_weight)
          - Add edge: composite -> canonical protein  (weight = 1.0)
          - Set node attribute 'name' = composite ID
          - Set node attribute 'category' = 'composite'

        The canonical protein node and its existing edges are untouched.
        """
        composite_nodes = []

        for (gene_symbol, cell_type), info in filtered.items():
            uniprot_id = info["uniprot_id"]
            raw_tstat = info["raw_tstat"]
            norm_weight = info["norm_weight"]

            # Composite node ID — safe for use as a graph node key
            cell_type_safe = cell_type.replace("/", "_").replace(" ", "_")
            composite_id = f"COMPOSITE:{gene_symbol}|{cell_type_safe}"

            # Add the composite node
            G.add_node(
                composite_id,
                name=composite_id,
                category="composite",
                gene=gene_symbol,
                cell_type=cell_type,
                uniprot_id=uniprot_id,
                raw_tstat=raw_tstat,
                norm_weight=norm_weight,
            )

            # Edge: Castleman -> composite (t-stat weight)
            G.add_edge(
                self.castleman_id,
                composite_id,
                weight=norm_weight,
            )

            # Edge: composite -> canonical protein (neutral weight)
            G.add_edge(
                composite_id,
                uniprot_id,
                weight=1.0,
            )

            composite_nodes.append(CompositeNodeInfo(
                node_id=composite_id,
                gene_symbol=gene_symbol,
                cell_type=cell_type,
                uniprot_id=uniprot_id,
                raw_tstat=raw_tstat,
                norm_weight=norm_weight,
            ))

        logger.info(f"Inserted {len(composite_nodes)} composite nodes")
        return composite_nodes

    @staticmethod
    def make_composite_id(gene_symbol: str, cell_type: str) -> str:
        """
        Generate the composite node ID for a (gene, cell_type) pair.
        Public static method so other modules can reference the same format.
        """
        cell_type_safe = cell_type.replace("/", "_").replace(" ", "_")
        return f"COMPOSITE:{gene_symbol}|{cell_type_safe}"
