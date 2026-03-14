"""
finetuner.py
------------
Purpose: Train GATConv on the composite-node-augmented graph.

What this does (plain English):
  Phase 5.1 failed because edge weights had no gradient signal — the model
  never learned what "high weight = important" means.

  Phase 5.2 fixes this structurally:
    1. The composite nodes (e.g. [TNF|Monocytes]) are NEW nodes in the graph.
       The model MUST route through them to reach TNF from Castleman.
    2. The edge from Castleman to each composite node carries a normalized
       t-stat weight. During training, the model learns that high-weight
       paths (strongly upregulated genes) correlate with drug efficacy.
    3. Siltuximab and tocilizumab are added as EXTRA positive supervision
       pairs for Castleman. The model learns: "these drugs work for Castleman."
    4. Adalimumab is NOT in supervision — it should emerge from the TNF
       pathway because TNF's composite nodes carry high t-stat weights.

  Success criterion:
    Adalimumab should rank significantly higher for Castleman than for
    non-TNF diseases (diabetes, hypertension, Alzheimer).

Usage:
  finetuner = CompositeFinetuner()
  results = finetuner.run(
      graph_path=Path("data/processed/full_graph.pkl"),
      csv_path=Path("data/experimental/iMCD_TAFRO_cell_specific_tstats.csv"),
      seeds=[42, 123, 303],
      epochs=200,
  )
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    """Results for one training seed."""
    seed: int
    castleman_rank: Optional[int]
    castleman_score: Optional[float]
    non_tnf_ranks: Dict[str, Optional[int]]   # disease_name -> rank


@dataclass
class FinetuneResult:
    """Aggregated results across all seeds."""
    seed_results: List[SeedResult] = field(default_factory=list)
    castleman_mean_rank: Optional[float] = None
    castleman_std_rank: float = 0.0
    non_tnf_mean_ranks: Dict[str, float] = field(default_factory=dict)
    disease_specific: Optional[bool] = None   # True if adalimumab ranks better for Castleman

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Phase 5.2 Fine-Tune Results",
            "=" * 60,
            f"Adalimumab rank for Castleman: "
            f"{self.castleman_mean_rank:,.0f} "
            f"(std={self.castleman_std_rank:.0f})"
            if self.castleman_mean_rank else "Castleman rank: N/A",
        ]
        for disease, rank in self.non_tnf_mean_ranks.items():
            lines.append(f"  Non-TNF {disease}: #{rank:,.0f}")
        verdict = "DISEASE-SPECIFIC" if self.disease_specific else "NOT DISEASE-SPECIFIC"
        lines.append(f"Verdict: {verdict}")
        lines.append("=" * 60)
        return "\n".join(lines)


# Diseases used to test disease-specificity
NON_TNF_DISEASES = {
    "MONDO:0005148": "Type 2 Diabetes",
    "MONDO:0005044": "Hypertension",
    "MONDO:0011382": "Alzheimer Disease",
}


class CompositeFinetuner:
    """
    Orchestrates the Phase 5.2 training pipeline:
      1. Build composite-node-augmented graph
      2. Add supervision pairs for siltuximab + tocilizumab
      3. Train GATConv
      4. Evaluate adalimumab rank

    Args:
        castleman_id:   Castleman disease CURIE
        adalimumab_id:  Adalimumab CHEMBL CURIE (held out from supervision)
        siltuximab_id:  Siltuximab CHEMBL CURIE (supervision positive)
        tocilizumab_id: Tocilizumab CHEMBL CURIE (supervision positive)
        min_tstat:      Minimum t-stat for composite node creation
    """

    CASTLEMAN_ID   = "MONDO:0015564"
    ADALIMUMAB_ID  = "CHEMBL.COMPOUND:CHEMBL1201580"
    SILTUXIMAB_ID  = "CHEMBL.COMPOUND:CHEMBL1743070"
    TOCILIZUMAB_ID = "CHEMBL.COMPOUND:CHEMBL1237022"

    def __init__(self, min_tstat: float = 2.0):
        self.min_tstat = min_tstat
        # Optional override for training data path — used in unit tests
        # to inject synthetic RepoDB files without needing the real Sockeye data.
        # Set this before calling run() if you need to override.
        self._training_data_path_override: Optional[Path] = None

    def run(
        self,
        graph_path: Path,
        csv_path: Path,
        seeds: List[int],
        epochs: int = 200,
        lr: float = 0.01,
        results_dir: Optional[Path] = None,
    ) -> FinetuneResult:
        """
        Full pipeline: build graph -> train -> evaluate, across multiple seeds.

        Args:
            graph_path:   Path to full_graph.pkl
            csv_path:     Path to iMCD_TAFRO_cell_specific_tstats.csv
            seeds:        List of random seeds for reproducibility
            epochs:       Training epochs per seed
            lr:           Learning rate
            results_dir:  If provided, saves per-seed JSON results here

        Returns:
            FinetuneResult with aggregated rankings across seeds
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from enhanced_kgnn.gene_mapper import GeneMapper
        from enhanced_kgnn.composite_node_builder import CompositeNodeBuilder
        from enhanced_kgnn.gat_predictor import GATPredictor, GATModel

        # ------------------------------------------------------------------
        # Step 1: Build composite-node-augmented graph (done once, reused)
        # ------------------------------------------------------------------
        logger.info("Step 1: Building composite-node-augmented graph")
        predictor = GATPredictor()

        # Apply training data path override if set (used in unit tests)
        if self._training_data_path_override is not None:
            predictor.training_data_path = self._training_data_path_override

        import pickle
        with open(graph_path, "rb") as f:
            nx_graph = pickle.load(f)

        mapper = GeneMapper(graph_path=graph_path)
        mapper.load_graph(nx_graph=nx_graph)

        builder = CompositeNodeBuilder(
            castleman_id=self.CASTLEMAN_ID,
            min_tstat=self.min_tstat,
        )
        build_result = builder.build(nx_graph, mapper, csv_path)
        augmented_graph = build_result.graph

        logger.info(build_result.summary())
        logger.info(
            f"Composite nodes created: {build_result.n_composite_nodes}"
        )

        # ------------------------------------------------------------------
        # Step 2: Build PyG Data from augmented graph
        # (predictor.nx_graph must point to the augmented graph)
        # ------------------------------------------------------------------
        logger.info("Step 2: Building PyG Data from augmented graph")
        predictor.nx_graph = augmented_graph
        predictor.processed_graph_path = graph_path  # keeps path reference valid

        # Rebuild entity index from augmented graph
        entities = sorted(augmented_graph.nodes())
        predictor.entity_to_idx = {e: i for i, e in enumerate(entities)}
        predictor.idx_to_entity = {i: e for e, i in predictor.entity_to_idx.items()}

        # Build Data using the augmented graph's edge weights
        data = predictor.build_graph(disease_edge_weights=None)

        # ------------------------------------------------------------------
        # Step 3: Add siltuximab + tocilizumab as extra positive supervision
        # These are known Castleman treatments. Adalimumab is held out.
        # ------------------------------------------------------------------
        logger.info("Step 3: Adding supervision pairs for Castleman treatments")
        castleman_idx = predictor.entity_to_idx.get(self.CASTLEMAN_ID)
        silt_idx = predictor.entity_to_idx.get(self.SILTUXIMAB_ID)
        toci_idx = predictor.entity_to_idx.get(self.TOCILIZUMAB_ID)
        adal_idx = predictor.entity_to_idx.get(self.ADALIMUMAB_ID)

        if castleman_idx is None:
            raise ValueError(f"Castleman ID not in graph: {self.CASTLEMAN_ID}")

        extra_edges = []
        extra_labels = []

        for drug_idx, drug_name in [
            (silt_idx, "siltuximab"),
            (toci_idx, "tocilizumab"),
        ]:
            if drug_idx is not None:
                extra_edges.append([drug_idx, castleman_idx])
                extra_labels.append(1)  # positive: known treatment
                logger.info(f"  Added supervision: {drug_name} -> Castleman (positive)")
            else:
                logger.warning(f"  {drug_name} not found in augmented graph")

        if adal_idx is not None:
            logger.info(
                f"  Adalimumab index={adal_idx} — held out, NOT in supervision"
            )

        if extra_edges:
            extra_edge_tensor = torch.tensor(extra_edges, dtype=torch.long)
            extra_label_tensor = torch.tensor(extra_labels, dtype=torch.float)

            # Append to existing train supervision
            data.train_edge_index = torch.cat(
                [data.train_edge_index, extra_edge_tensor.t()], dim=1
            )
            data.train_edge_labels = torch.cat(
                [data.train_edge_labels, extra_label_tensor]
            )

        # ------------------------------------------------------------------
        # Step 4: Train across seeds and evaluate
        # ------------------------------------------------------------------
        logger.info("Step 4: Training and evaluating")

        input_dim = data.x.shape[1]
        seed_results: List[SeedResult] = []

        for seed in seeds:
            logger.info(f"\n{'='*50}")
            logger.info(f"Seed {seed}")
            logger.info(f"{'='*50}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            model = GATModel(
                input_dim=input_dim,
                hidden_dim=64,
                output_dim=32,
                heads=4,
                edge_dim=1,
                dropout=0.1,
            )
            model = predictor.train_model(data, model, epochs=epochs, lr=lr)

            # Evaluate adalimumab rank for Castleman
            cast_rank, cast_score = predictor.get_adalimumab_rank(
                model, data, self.CASTLEMAN_ID
            )
            logger.info(
                f"Adalimumab rank for Castleman: "
                f"{'#'+str(cast_rank) if cast_rank else 'NOT FOUND'}"
            )

            # Evaluate for non-TNF diseases
            non_tnf_ranks: Dict[str, Optional[int]] = {}
            for disease_id, disease_name in NON_TNF_DISEASES.items():
                rank, _ = predictor.get_adalimumab_rank(model, data, disease_id)
                non_tnf_ranks[disease_name] = rank
                status = f"#{rank:,}" if rank else "NOT FOUND"
                logger.info(f"  Non-TNF {disease_name}: {status}")

            seed_results.append(SeedResult(
                seed=seed,
                castleman_rank=cast_rank,
                castleman_score=float(cast_score) if cast_score is not None else None,
                non_tnf_ranks=non_tnf_ranks,
            ))

        # ------------------------------------------------------------------
        # Step 5: Aggregate across seeds
        # ------------------------------------------------------------------
        castleman_ranks = [
            r.castleman_rank for r in seed_results
            if r.castleman_rank is not None
        ]
        castleman_mean = float(np.mean(castleman_ranks)) if castleman_ranks else None
        castleman_std  = float(np.std(castleman_ranks, ddof=1)) if len(castleman_ranks) > 1 else 0.0

        non_tnf_mean_ranks: Dict[str, float] = {}
        for disease_name in NON_TNF_DISEASES.values():
            ranks = [
                r.non_tnf_ranks[disease_name]
                for r in seed_results
                if r.non_tnf_ranks.get(disease_name) is not None
            ]
            if ranks:
                non_tnf_mean_ranks[disease_name] = float(np.mean(ranks))

        # Disease-specific: adalimumab mean rank for Castleman < mean rank for non-TNF
        disease_specific = None
        if castleman_mean and non_tnf_mean_ranks:
            overall_non_tnf = np.mean(list(non_tnf_mean_ranks.values()))
            disease_specific = castleman_mean < overall_non_tnf

        result = FinetuneResult(
            seed_results=seed_results,
            castleman_mean_rank=castleman_mean,
            castleman_std_rank=castleman_std,
            non_tnf_mean_ranks=non_tnf_mean_ranks,
            disease_specific=disease_specific,
        )

        logger.info(result.summary())

        # Save results if directory provided
        if results_dir:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            output = {
                "castleman_mean_rank": castleman_mean,
                "castleman_std_rank": castleman_std,
                "non_tnf_mean_ranks": non_tnf_mean_ranks,
                "disease_specific": bool(disease_specific) if disease_specific is not None else None,
                "n_composite_nodes": build_result.n_composite_nodes,
                "min_tstat": self.min_tstat,
                "seeds": seeds,
                "epochs": epochs,
                "seed_results": [
                    {
                        "seed": r.seed,
                        "castleman_rank": r.castleman_rank,
                        "castleman_score": r.castleman_score,
                        "non_tnf_ranks": r.non_tnf_ranks,
                    }
                    for r in seed_results
                ],
            }
            out_path = results_dir / "phase_5_2_results.json"
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {out_path}")

        return result
