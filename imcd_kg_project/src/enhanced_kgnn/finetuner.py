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

  Phase 5.5 addition:
    After training, we save two things per seed:
      a) top_500_castleman — the 500 highest-scoring drugs for Castleman
         (whatever the model rates highest, including unknown compounds)
      b) drug_ranks_castleman — explicit rank + score for every drug in
         DRUGS_OF_INTEREST regardless of where they sit in the full ranking.
         This is how we answer the professor's question: "where does
         siltuximab rank? where does adalimumab rank?"
    Both come from a single forward pass via get_drug_ranks().

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
    castleman_rank: Optional[int]         # adalimumab rank for Castleman (for aggregation)
    castleman_score: Optional[float]
    non_tnf_ranks: Dict[str, Optional[int]]   # disease_name -> adalimumab rank
    top_500_castleman: List[Dict] = field(default_factory=list)
    # Top-500 highest-scoring drugs for Castleman.
    # [{rank, drug_id, name, score}]
    # Shows whatever the model rates highest — includes unknown compounds.
    drug_ranks_castleman: Dict[str, Dict] = field(default_factory=dict)
    # Explicit rank + score for confirmed drugs of interest, regardless of rank.
    # {drug_name: {rank, score, mechanism, drug_id}}


@dataclass
class FinetuneResult:
    """Aggregated results across all seeds."""
    seed_results: List[SeedResult] = field(default_factory=list)
    castleman_mean_rank: Optional[float] = None
    castleman_std_rank: float = 0.0
    non_tnf_mean_ranks: Dict[str, float] = field(default_factory=dict)
    disease_specific: Optional[bool] = None
    drug_mean_ranks_castleman: Dict[str, Dict] = field(default_factory=dict)
    # Mean rank + score across seeds for each confirmed drug of interest.
    # {drug_name: {mean_rank, std_rank, mean_score, mechanism, drug_id}}

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
        if self.drug_mean_ranks_castleman:
            lines.append("Drug ranks for Castleman (confirmed drugs of interest):")
            for name, info in sorted(
                self.drug_mean_ranks_castleman.items(),
                key=lambda x: x[1]["mean_rank"]
            ):
                lines.append(
                    f"  {name:20s}  rank=#{info['mean_rank']:,.0f} "
                    f"(std={info['std_rank']:.0f})  "
                    f"score={info['mean_score']:.6f}  [{info['mechanism']}]"
                )
        lines.append("=" * 60)
        return "\n".join(lines)


# Diseases used to test disease-specificity
NON_TNF_DISEASES = {
    "MONDO:0005148": "Type 2 Diabetes",
    "MONDO:0005044": "Hypertension",
    "MONDO:0011382": "Alzheimer Disease",
}

# Confirmed drugs of interest — CHEMBL IDs verified against the RTX-KG2 graph.
# See CLAUDE.md: adalimumab, siltuximab, tocilizumab confirmed by entity name
# search on Sockeye.
#
# To add more drugs: verify the CHEMBL ID exists in the graph by checking
# graph node names, then add here. Do NOT add unverified IDs.
DRUGS_OF_INTEREST: Dict[str, Dict[str, str]] = {
    "CHEMBL.COMPOUND:CHEMBL1201580": {"name": "adalimumab",  "mechanism": "Anti-TNF"},
    "CHEMBL.COMPOUND:CHEMBL1743070": {"name": "siltuximab",  "mechanism": "Anti-IL-6"},
    "CHEMBL.COMPOUND:CHEMBL1237022": {"name": "tocilizumab", "mechanism": "Anti-IL-6R"},
}


class CompositeFinetuner:
    """
    Orchestrates the composite-node fine-tuning pipeline:
      1. Build composite-node-augmented graph
      2. Add supervision pairs for siltuximab + tocilizumab
      3. Train GATConv across multiple seeds
      4. Evaluate: adalimumab rank + top-500 drug table + drug ranks for
         confirmed drugs of interest (all from a single forward pass)

    Args:
        min_tstat: Minimum t-stat for composite node creation
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
        cell_types: Optional[List[str]] = None,
    ) -> FinetuneResult:
        """
        Full pipeline: build graph -> train -> evaluate, across multiple seeds.

        Args:
            graph_path:   Path to full_graph.pkl
            csv_path:     Path to iMCD_TAFRO_cell_specific_tstats.csv
            seeds:        List of random seeds for reproducibility
            epochs:       Training epochs per seed
            lr:           Learning rate
            results_dir:  If provided, saves results JSON here
            cell_types:   Optional list of cell type column names to use.
                          None (default) uses all five cell types.
                          Example: ["T cells"] for a T-cells-only experiment.

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
        build_result = builder.build(nx_graph, mapper, csv_path, cell_types=cell_types)
        augmented_graph = build_result.graph

        logger.info(build_result.summary())
        logger.info(f"Composite nodes created: {build_result.n_composite_nodes}")

        # ------------------------------------------------------------------
        # Step 2: Build PyG Data from augmented graph
        # ------------------------------------------------------------------
        logger.info("Step 2: Building PyG Data from augmented graph")
        predictor.nx_graph = augmented_graph
        predictor.processed_graph_path = graph_path

        entities = sorted(augmented_graph.nodes())
        predictor.entity_to_idx = {e: i for i, e in enumerate(entities)}
        predictor.idx_to_entity = {i: e for e, i in predictor.entity_to_idx.items()}

        data = predictor.build_graph(disease_edge_weights=None)

        # ------------------------------------------------------------------
        # Step 3: Add siltuximab + tocilizumab as extra positive supervision
        # ------------------------------------------------------------------
        logger.info("Step 3: Adding supervision pairs for Castleman treatments")
        castleman_idx = predictor.entity_to_idx.get(self.CASTLEMAN_ID)
        silt_idx      = predictor.entity_to_idx.get(self.SILTUXIMAB_ID)
        toci_idx      = predictor.entity_to_idx.get(self.TOCILIZUMAB_ID)
        adal_idx      = predictor.entity_to_idx.get(self.ADALIMUMAB_ID)

        if castleman_idx is None:
            raise ValueError(f"Castleman ID not in graph: {self.CASTLEMAN_ID}")

        extra_edges, extra_labels = [], []
        for drug_idx, drug_name in [(silt_idx, "siltuximab"), (toci_idx, "tocilizumab")]:
            if drug_idx is not None:
                extra_edges.append([drug_idx, castleman_idx])
                extra_labels.append(1)
                logger.info(f"  Added supervision: {drug_name} -> Castleman (positive)")
            else:
                logger.warning(f"  {drug_name} not found in augmented graph")

        if adal_idx is not None:
            logger.info(f"  Adalimumab index={adal_idx} — held out, NOT in supervision")

        if extra_edges:
            data.train_edge_index = torch.cat(
                [data.train_edge_index,
                 torch.tensor(extra_edges, dtype=torch.long).t()], dim=1
            )
            data.train_edge_labels = torch.cat(
                [data.train_edge_labels,
                 torch.tensor(extra_labels, dtype=torch.float)]
            )

        # ------------------------------------------------------------------
        # Step 4: Train across seeds and evaluate
        # ------------------------------------------------------------------
        logger.info("Step 4: Training and evaluating")

        doi_ids = {k: v["name"] for k, v in DRUGS_OF_INTEREST.items()}
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

            # Single forward pass for Castleman:
            #   - top_500: whatever the model rates highest (table for professor)
            #   - specific_ranks: explicit rank for adalimumab, siltuximab,
            #     tocilizumab regardless of their position in the full ranking
            top_500, specific_ranks = predictor.get_drug_ranks(
                model=model,
                data=data,
                disease_entity=self.CASTLEMAN_ID,
                drugs_of_interest=doi_ids,
                top_n=500,
            )

            # Annotate specific_ranks with mechanism + build drug_ranks_castleman
            drug_ranks_castleman: Dict[str, Dict] = {}
            for drug_id, info in specific_ranks.items():
                doi = DRUGS_OF_INTEREST[drug_id]
                drug_ranks_castleman[doi["name"]] = {
                    "rank":      info["rank"],
                    "score":     info["score"],
                    "mechanism": doi["mechanism"],
                    "drug_id":   drug_id,
                }

            # Adalimumab rank for Castleman — comes from specific_ranks, no extra pass
            adal_info  = specific_ranks.get(self.ADALIMUMAB_ID, {})
            cast_rank  = adal_info.get("rank")
            cast_score = adal_info.get("score")
            logger.info(
                f"Adalimumab rank for Castleman: "
                f"{'#'+str(cast_rank) if cast_rank else 'NOT FOUND'}"
            )
            for doi_name, doi_info in sorted(
                drug_ranks_castleman.items(), key=lambda x: x[1]["rank"]
            ):
                logger.info(
                    f"  {doi_name}: rank=#{doi_info['rank']:,}  "
                    f"score={doi_info['score']:.6f}  [{doi_info['mechanism']}]"
                )

            # Adalimumab rank for non-TNF diseases (disease-specificity check)
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
                top_500_castleman=top_500,
                drug_ranks_castleman=drug_ranks_castleman,
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

        disease_specific = None
        if castleman_mean and non_tnf_mean_ranks:
            overall_non_tnf = np.mean(list(non_tnf_mean_ranks.values()))
            disease_specific = castleman_mean < overall_non_tnf

        # Aggregate drug ranks across seeds
        drug_mean_ranks_castleman: Dict[str, Dict] = {}
        all_drug_names: set = set()
        for r in seed_results:
            all_drug_names.update(r.drug_ranks_castleman.keys())

        for drug_name in sorted(all_drug_names):
            seed_data = [
                r.drug_ranks_castleman[drug_name]
                for r in seed_results
                if drug_name in r.drug_ranks_castleman
            ]
            ranks  = [d["rank"]  for d in seed_data if d.get("rank")  is not None]
            scores = [d["score"] for d in seed_data if d.get("score") is not None]
            if ranks:
                drug_mean_ranks_castleman[drug_name] = {
                    "mean_rank":  float(np.mean(ranks)),
                    "std_rank":   float(np.std(ranks, ddof=1)) if len(ranks) > 1 else 0.0,
                    "mean_score": float(np.mean(scores)),
                    "mechanism":  seed_data[0].get("mechanism", ""),
                    "drug_id":    seed_data[0].get("drug_id", ""),
                }

        result = FinetuneResult(
            seed_results=seed_results,
            castleman_mean_rank=castleman_mean,
            castleman_std_rank=castleman_std,
            non_tnf_mean_ranks=non_tnf_mean_ranks,
            disease_specific=disease_specific,
            drug_mean_ranks_castleman=drug_mean_ranks_castleman,
        )

        logger.info(result.summary())

        # ------------------------------------------------------------------
        # Step 6: Save results
        # ------------------------------------------------------------------
        if results_dir:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            output = {
                "castleman_mean_rank":       castleman_mean,
                "castleman_std_rank":        castleman_std,
                "non_tnf_mean_ranks":        non_tnf_mean_ranks,
                "disease_specific":          bool(disease_specific) if disease_specific is not None else None,
                "n_composite_nodes":         build_result.n_composite_nodes,
                "min_tstat":                 self.min_tstat,
                "cell_types":               cell_types,
                "seeds":                     seeds,
                "epochs":                    epochs,
                "drug_mean_ranks_castleman": drug_mean_ranks_castleman,
                "seed_results": [
                    {
                        "seed":                 r.seed,
                        "castleman_rank":       r.castleman_rank,
                        "castleman_score":      r.castleman_score,
                        "non_tnf_ranks":        r.non_tnf_ranks,
                        "drug_ranks_castleman": r.drug_ranks_castleman,
                        "top_500_castleman":    r.top_500_castleman,
                    }
                    for r in seed_results
                ],
            }
            out_path = results_dir / "phase_5_2_results.json"
            with open(out_path, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {out_path}")

        return result
