#!/bin/bash
#SBATCH --job-name=phase_1_2
#SBATCH --account=st-singha53-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/phase_1_2_%j.txt

echo "Phase 1.2: Load Full Graph & Validate"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================"

module load gcc/9.4.0 miniconda3/4.9.2
source activate imcd_kg

cd /scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project
export KG_ROOT="/arc/project/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project/data/kgml_data/bkg_rtxkg2c_v2.7.3"

python - << 'PY'
import os, sys, json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

from enhanced_kgnn.rtx_kg_loader import RTXKGLoader
from config import PROCESSED_GRAPH_PATH, ADALIMUMAB_ID, CASTLEMAN_ID, TNF_ID

print("="*70)
print("PHASE 1.2: LOADING FULL RTX-KG2 GRAPH")
print("="*70)

# Initialize loader
kg_path = Path(os.environ["KG_ROOT"])
loader = RTXKGLoader(kg_path)

# Define entity types to load (focus on relevant entities)
entity_filter = [
    'CHEMBL.COMPOUND',  # Drugs
    'MONDO',            # Diseases
    'UniProtKB',        # Proteins
    'NCBIGene',         # Genes
    'HGNC',             # Gene names
]

print(f"\nFiltering for entity types: {entity_filter}")

# Load graph
graph = loader.load_full_graph(entity_types_filter=entity_filter)

print("\n" + "="*70)
print("VALIDATING CRITICAL ENTITIES AND PATHS")
print("="*70)

validation = loader.validate_critical_entities(
    graph, ADALIMUMAB_ID, CASTLEMAN_ID, TNF_ID
)

print("\nValidation Results:")
for key, value in validation.items():
    status = "✓" if value else "✗"
    print(f"  {status} {key}: {value}")

# Save validation results
results_dir = Path("results/phase_1_2")
results_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / "validation.json", 'w') as f:
    json.dump(validation, f, indent=2)

print(f"\n✓ Validation saved to: {results_dir / 'validation.json'}")

# Save graph for future use
print("\n" + "="*70)
print("SAVING GRAPH FOR FUTURE PHASES")
print("="*70)

loader.save_graph(graph, PROCESSED_GRAPH_PATH)

print("\n" + "="*70)
print("✓ PHASE 1.2 COMPLETE")
print("="*70)
print(f"Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
print(f"Saved to: {PROCESSED_GRAPH_PATH}")
print(f"Complete pathway: {validation['complete_pathway']}")

sys.exit(0)

PY

echo "================================"
echo "End: $(date)"