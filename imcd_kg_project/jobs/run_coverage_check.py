"""
run_coverage_check.py
---------------------
Checks how many of the 12,500 genes in the iMCD-TAFRO scRNA-seq CSV
map to UniProtKB nodes in the RTX-KG2 graph (full_graph.pkl).

Run on Sockeye:
  conda run -n imcd_kg python jobs/run_coverage_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enhanced_kgnn.gene_mapper import GeneMapper

GRAPH_PATH = Path("data/processed/full_graph.pkl")
CSV_PATH = Path("data/experimental/iMCD_TAFRO_cell_specific_tstats.csv")

print("Loading graph (this takes ~1-2 minutes)...")
mapper = GeneMapper(graph_path=GRAPH_PATH)
mapper.load_graph()

print("Mapping CSV gene symbols to UniProtKB IDs...")
report = mapper.map_csv(CSV_PATH)
print(report.summary())

print("\nFirst 20 unmatched genes:")
for g in report.unmatched[:20]:
    print(f"  {g}")

print("\nSample matched genes (first 10):")
for gene, uid in list(report.matched.items())[:10]:
    print(f"  {gene:15} -> {uid}")

# Also check our key genes specifically
key_genes = ["TNF", "IL6", "STAT3", "IL10", "IL8", "IL4", "CD4"]
print("\nKey gene checks:")
for gene in key_genes:
    uid = mapper.get_uniprot_id(gene)
    status = uid if uid else "NOT FOUND IN GRAPH"
    print(f"  {gene:10} -> {status}")
