"""
phase_5_2_diagnostics.py
------------------------
One-shot diagnostic script for Phase 5.2 composite node pipeline.
Runs every check needed before writing any model code.

Checks:
  1. Supervision drugs (siltuximab, tocilizumab) in graph + edges to Castleman
  2. Castleman protein neighbors - how many, which ones
  3. CSV gene coverage against those neighbors
  4. T-stat values for Castleman protein neighbors across all cell types
  5. Top 10 genes per cell type by t-stat (composite node candidates)
  6. Key path check: Adalimumab -> TNF -> Castleman still intact

Run on Sockeye:
  conda run -n imcd_kg python jobs/phase_5_2_diagnostics.py
"""
import csv
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from enhanced_kgnn.gene_mapper import GeneMapper

GRAPH_PATH  = Path("data/processed/full_graph.pkl")
CSV_PATH    = Path("data/experimental/iMCD_TAFRO_cell_specific_tstats.csv")
CASTLEMAN   = "MONDO:0015564"
TNF         = "UniProtKB:P01375"
ADALIMUMAB  = "CHEMBL.COMPOUND:CHEMBL1201580"
SILTUXIMAB  = "CHEMBL.COMPOUND:CHEMBL1615786"
TOCILIZUMAB = "CHEMBL.COMPOUND:CHEMBL1201823"
SEP = "=" * 65

print(SEP)
print("PHASE 5.2 PRE-BUILD DIAGNOSTICS")
print(SEP)

# Load graph once — reused across all checks
print("\nLoading graph (1-2 min)...")
with open(GRAPH_PATH, "rb") as f:
    G = pickle.load(f)
print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Load mapper (reuses already-loaded graph, no second disk read)
mapper = GeneMapper(graph_path=GRAPH_PATH)
mapper.load_graph(nx_graph=G)

# ------------------------------------------------------------------
# CHECK 1: Supervision drugs
# ------------------------------------------------------------------
print(f"\n{SEP}")
print("CHECK 1: Supervision drugs")
print(SEP)
for name, uid in [("siltuximab", SILTUXIMAB),
                  ("tocilizumab", TOCILIZUMAB),
                  ("adalimumab", ADALIMUMAB)]:
    in_g = uid in G
    edge = G.has_edge(uid, CASTLEMAN) if in_g else False
    label = G.nodes[uid].get("name", "?") if in_g else "NOT IN GRAPH"
    print(f"  {name:<15} in_graph={str(in_g):<6} "
          f"edge_to_castleman={str(edge):<6} name='{label}'")

# ------------------------------------------------------------------
# CHECK 2: Castleman protein neighbors
# ------------------------------------------------------------------
print(f"\n{SEP}")
print("CHECK 2: Castleman protein neighbors")
print(SEP)
castleman_proteins = [
    n for n in G.neighbors(CASTLEMAN)
    if n.startswith("UniProtKB:")
]
print(f"  Total protein neighbors: {len(castleman_proteins)}")
print(f"  TNF is a neighbor:       {TNF in castleman_proteins}")

# ------------------------------------------------------------------
# CHECK 3: How many Castleman protein neighbors are in the CSV
# ------------------------------------------------------------------
print(f"\n{SEP}")
print("CHECK 3: Castleman protein neighbors covered by CSV")
print(SEP)

neighbor_name_to_uid = {}
for uid in castleman_proteins:
    name = G.nodes[uid].get("name", "")
    if name:
        neighbor_name_to_uid[name.upper()] = uid

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    cell_types = [h for h in (reader.fieldnames or []) if h not in ("", "gene")]
    csv_rows = {row["gene"].strip().upper(): row for row in reader}

matched_neighbors = {
    sym: uid for sym, uid in neighbor_name_to_uid.items()
    if sym in csv_rows
}
print(f"  Castleman neighbors with t-stats in CSV: "
      f"{len(matched_neighbors)}/{len(castleman_proteins)}")
print(f"  Cell types: {cell_types}")

# ------------------------------------------------------------------
# CHECK 4: T-stats for matched Castleman neighbors
# ------------------------------------------------------------------
print(f"\n{SEP}")
print("CHECK 4: T-stats for Castleman protein neighbors in CSV")
print(SEP)
header = f"  {'Gene':<12} " + "  ".join(f"{ct[:10]:>10}" for ct in cell_types)
print(header)
print("  " + "-" * (len(header) - 2))
for gene_sym, uid in sorted(matched_neighbors.items()):
    row = csv_rows.get(gene_sym, {})
    vals = []
    for ct in cell_types:
        try:
            vals.append(float(row.get(ct, 0)))
        except ValueError:
            vals.append(0.0)
    val_str = "  ".join(f"{v:>10.2f}" for v in vals)
    print(f"  {gene_sym:<12} {val_str}")

# ------------------------------------------------------------------
# CHECK 5: Top 10 genes per cell type (composite node candidates)
# ------------------------------------------------------------------
print(f"\n{SEP}")
print("CHECK 5: Top 10 genes per cell type by t-stat (graph-matched only)")
print(SEP)
mapped_data = mapper.map_csv_with_tstats(CSV_PATH, min_tstat=0.0)
for ct in cell_types:
    entries = [
        (gene, info[ct])
        for gene, info in mapped_data.items()
        if ct in info and info[ct] > 0
    ]
    entries.sort(key=lambda x: -x[1])
    print(f"\n  {ct} — top 10:")
    for gene, tstat in entries[:10]:
        uid = mapped_data[gene]["uniprot_id"]
        print(f"    {gene:<12} t={tstat:>8.2f}  {uid}")

# ------------------------------------------------------------------
# CHECK 6: Key path integrity
# ------------------------------------------------------------------
print(f"\n{SEP}")
print("CHECK 6: Key path integrity")
print(SEP)
ada_tnf  = G.has_edge(ADALIMUMAB, TNF)
tnf_cast = G.has_edge(TNF, CASTLEMAN)
print(f"  Adalimumab -> TNF:    {ada_tnf}")
print(f"  TNF -> Castleman:     {tnf_cast}")
print(f"  Full path intact:     {ada_tnf and tnf_cast}")

print(f"\n{SEP}")
print("DIAGNOSTICS COMPLETE")
print(SEP)
