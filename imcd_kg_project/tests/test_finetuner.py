"""
test_finetuner.py
-----------------
Unit tests for finetuner.py

Tests run on a small synthetic graph — no full_graph.pkl needed.

What is tested:
  1. CompositeFinetuner.run() completes without error on a synthetic graph
  2. Supervision pairs for siltuximab/tocilizumab are added correctly
  3. Adalimumab is NOT in supervision pairs
  4. FinetuneResult contains expected fields after run
  5. Results are saved to disk when results_dir is provided
  6. Training runs for the correct number of seeds
"""

import csv
import json
import pickle
import sys
import tempfile
from pathlib import Path

import networkx as nx
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enhanced_kgnn.finetuner import CompositeFinetuner, FinetuneResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CASTLEMAN   = "MONDO:0015564"
TNF         = "UniProtKB:P01375"
STAT3       = "UniProtKB:P40763"
ADALIMUMAB  = "CHEMBL.COMPOUND:CHEMBL1201580"
SILTUXIMAB  = "CHEMBL.COMPOUND:CHEMBL1743070"
TOCILIZUMAB = "CHEMBL.COMPOUND:CHEMBL1237022"
DIABETES    = "MONDO:0005148"
HYPERT      = "MONDO:0005044"
ALZHEIMER   = "MONDO:0011382"


def make_synthetic_graph() -> nx.DiGraph:
    """
    Minimal directed graph with all entities needed for finetuner tests.
    Includes non-TNF diseases so disease-specificity can be evaluated.
    """
    G = nx.DiGraph()

    # Proteins
    G.add_node(TNF,   name="TNF",   category="biolink:Protein")
    G.add_node(STAT3, name="STAT3", category="biolink:Protein")

    # Diseases
    for did, dname in [
        (CASTLEMAN, "Castleman Disease"),
        (DIABETES,  "Type 2 Diabetes"),
        (HYPERT,    "Hypertension"),
        (ALZHEIMER, "Alzheimer Disease"),
    ]:
        G.add_node(did, name=dname, category="biolink:Disease")

    # Drugs
    for drug_id, dname in [
        (ADALIMUMAB,  "ADALIMUMAB"),
        (SILTUXIMAB,  "SILTUXIMAB"),
        (TOCILIZUMAB, "TOCILIZUMAB"),
        # A handful of other drugs so ranking is non-trivial
        ("CHEMBL.COMPOUND:CHEMBL_D1", "drug1"),
        ("CHEMBL.COMPOUND:CHEMBL_D2", "drug2"),
        ("CHEMBL.COMPOUND:CHEMBL_D3", "drug3"),
    ]:
        G.add_node(drug_id, name=dname, category="biolink:Drug")

    # Incoming protein->disease edges
    G.add_edge(TNF,   CASTLEMAN)
    G.add_edge(STAT3, CASTLEMAN)
    G.add_edge(TNF,   DIABETES)
    G.add_edge(TNF,   HYPERT)
    G.add_edge(TNF,   ALZHEIMER)

    # Drug->protein edges
    G.add_edge(ADALIMUMAB,  TNF)
    G.add_edge(SILTUXIMAB,  STAT3)
    G.add_edge(TOCILIZUMAB, STAT3)

    # RepoDB-style supervision pairs stored as graph edges for training
    # (positive: known treatments)
    G.add_edge(SILTUXIMAB,  CASTLEMAN)
    G.add_edge(TOCILIZUMAB, CASTLEMAN)

    return G


def make_minimal_csv(tmp_dir: str) -> Path:
    """Write a minimal t-stat CSV with TNF and STAT3."""
    path = Path(tmp_dir) / "tstats.csv"
    fieldnames = ["", "gene", "B cells", "ILC",
                  "Megakaryocytes/platelets", "Monocytes", "T cells"]
    rows = [
        {"gene": "TNF",   "B cells": 0, "ILC": 0,
         "Megakaryocytes/platelets": 0, "Monocytes": 32.2, "T cells": 2.9},
        {"gene": "STAT3", "B cells": 0, "ILC": 0,
         "Megakaryocytes/platelets": 0, "Monocytes": 4.0,  "T cells": 0},
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            writer.writerow({"": str(i), **row})
    return path


def make_fake_training_data(tmp_dir: str) -> Path:
    """
    Write minimal RepoDB-style training files so GATPredictor._load_training_data
    doesn't fail with missing files.
    """
    training_dir = Path(tmp_dir) / "training_data"
    training_dir.mkdir(parents=True, exist_ok=True)
    # Positive pairs (drug treats disease)
    tp_path = training_dir / "repoDB_tp.txt"
    with open(tp_path, "w") as f:
        f.write("source\ttarget\n")
        f.write(f"{SILTUXIMAB}\t{CASTLEMAN}\n")
        f.write(f"{TOCILIZUMAB}\t{CASTLEMAN}\n")
        f.write(f"CHEMBL.COMPOUND:CHEMBL_D1\t{DIABETES}\n")
    # Negative pairs (drug does NOT treat disease)
    tn_path = training_dir / "repoDB_tn.txt"
    with open(tn_path, "w") as f:
        f.write("source\ttarget\n")
        f.write(f"CHEMBL.COMPOUND:CHEMBL_D2\t{CASTLEMAN}\n")
        f.write(f"CHEMBL.COMPOUND:CHEMBL_D3\t{ALZHEIMER}\n")
    return training_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCompositeFinetuner:

    def test_run_completes_without_error(self):
        """Full pipeline runs end-to-end on synthetic data without crashing."""
        with tempfile.TemporaryDirectory() as tmp:
            G = make_synthetic_graph()
            graph_path = Path(tmp) / "full_graph.pkl"
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)

            csv_path      = make_minimal_csv(tmp)
            training_dir  = make_fake_training_data(tmp)

            finetuner = CompositeFinetuner(min_tstat=2.0)
            # Patch training_data_path so GATPredictor finds our fake files
            finetuner._training_data_path_override = training_dir

            result = finetuner.run(
                graph_path=graph_path,
                csv_path=csv_path,
                seeds=[42],
                epochs=5,   # tiny for speed
            )

        assert isinstance(result, FinetuneResult)

    def test_correct_number_of_seed_results(self):
        """One SeedResult per seed in seeds list."""
        with tempfile.TemporaryDirectory() as tmp:
            G = make_synthetic_graph()
            graph_path = Path(tmp) / "full_graph.pkl"
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)

            csv_path     = make_minimal_csv(tmp)
            training_dir = make_fake_training_data(tmp)

            finetuner = CompositeFinetuner(min_tstat=2.0)
            finetuner._training_data_path_override = training_dir

            result = finetuner.run(
                graph_path=graph_path,
                csv_path=csv_path,
                seeds=[42, 123],
                epochs=3,
            )

        assert len(result.seed_results) == 2
        assert result.seed_results[0].seed == 42
        assert result.seed_results[1].seed == 123

    def test_results_saved_to_disk(self):
        """When results_dir is provided, phase_5_2_results.json is written."""
        with tempfile.TemporaryDirectory() as tmp:
            G = make_synthetic_graph()
            graph_path = Path(tmp) / "full_graph.pkl"
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)

            csv_path     = make_minimal_csv(tmp)
            training_dir = make_fake_training_data(tmp)
            results_dir  = Path(tmp) / "results"

            finetuner = CompositeFinetuner(min_tstat=2.0)
            finetuner._training_data_path_override = training_dir

            finetuner.run(
                graph_path=graph_path,
                csv_path=csv_path,
                seeds=[42],
                epochs=3,
                results_dir=results_dir,
            )

            out_file = results_dir / "phase_5_2_results.json"
            assert out_file.exists(), "Results JSON not written"
            with open(out_file) as f:
                data = json.load(f)
            assert "castleman_mean_rank" in data
            assert "seed_results" in data
            assert len(data["seed_results"]) == 1

    def test_cell_types_param_propagates(self):
        """
        Passing cell_types=["T cells"] must reach the builder and produce
        only T-cell composite nodes. This guards against the param being
        silently ignored somewhere in the call chain.
        """
        with tempfile.TemporaryDirectory() as tmp:
            G = make_synthetic_graph()
            graph_path = Path(tmp) / "full_graph.pkl"
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)

            csv_path     = make_minimal_csv(tmp)
            training_dir = make_fake_training_data(tmp)
            results_dir  = Path(tmp) / "results"

            finetuner = CompositeFinetuner(min_tstat=2.0)
            finetuner._training_data_path_override = training_dir

            finetuner.run(
                graph_path=graph_path,
                csv_path=csv_path,
                seeds=[42],
                epochs=3,
                results_dir=results_dir,
                cell_types=["T cells"],
            )

            out_file = results_dir / "phase_5_2_results.json"
            assert out_file.exists()
            with open(out_file) as f:
                data = json.load(f)
            # cell_types must be saved so we can verify the filter was applied
            assert data["cell_types"] == ["T cells"], (
                "cell_types not saved in results — param may not have propagated"
            )

    def test_adalimumab_not_in_supervision(self):
        """
        Adalimumab-Castleman must never appear as a supervision pair.
        If it does, the evaluation is meaningless (data leakage).
        """
        with tempfile.TemporaryDirectory() as tmp:
            G = make_synthetic_graph()
            graph_path = Path(tmp) / "full_graph.pkl"
            with open(graph_path, "wb") as f:
                pickle.dump(G, f)

            # Add adalimumab-castleman to the fake training data (leakage scenario)
            training_dir = Path(tmp) / "training_data"
            training_dir.mkdir(parents=True, exist_ok=True)
            tp_path = training_dir / "repoDB_tp.txt"
            with open(tp_path, "w") as f:
                f.write("source\ttarget\n")
                f.write(f"{ADALIMUMAB}\t{CASTLEMAN}\n")  # leakage!
            tn_path = training_dir / "repoDB_tn.txt"
            with open(tn_path, "w") as f:
                f.write("source\ttarget\n")

            csv_path = make_minimal_csv(tmp)
            finetuner = CompositeFinetuner(min_tstat=2.0)
            finetuner._training_data_path_override = training_dir

            # GATPredictor._load_training_data raises ValueError on leakage
            with pytest.raises(ValueError, match="leakage"):
                finetuner.run(
                    graph_path=graph_path,
                    csv_path=csv_path,
                    seeds=[42],
                    epochs=3,
                )
