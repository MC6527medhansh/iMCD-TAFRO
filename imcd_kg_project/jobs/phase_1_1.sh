#!/bin/bash
#SBATCH --job-name=phase_1_1
#SBATCH --account=st-singha53-1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --output=logs/phase_1_1_%j.txt

echo "Phase 1.1: Entity Verification"
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "================================"

module load gcc/9.4.0 miniconda3/4.9.2
source activate imcd_kg

cd /scratch/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project
export KG_ROOT="/arc/project/st-singha53-1/mchoubey/iMCD-TAFRO/imcd_kg_project/data/kgml_data/bkg_rtxkg2c_v2.7.3"

python - << 'PY'
import os, sys
from pathlib import Path

KG = Path(os.environ["KG_ROOT"])
from src.enhanced_kgnn.rtx_kg_loader import RTXKGLoader

print(f"KG root: {KG}")
ldr = RTXKGLoader(KG)
print(f"Nodes file: {ldr.nodes_file} (exists={Path(ldr.nodes_file).exists()})")
print(f"Edges file: {ldr.edges_file} (exists={Path(ldr.edges_file).exists()})")

targets = {
    "adalimumab": "CHEMBL.COMPOUND:CHEMBL1201580",
    "castleman": "MONDO:0015564",
    "tnf_gene": "UniProtKB:P01375",
}

found = {k: None for k in targets}
hits = set(targets.values())

print("\nScanning nodes file for target IDs...")
with open(ldr.nodes_file, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f, 1):
        parts = line.rstrip("\n").split("\t")
        if not parts:
            continue
        node_id = parts[0]
        if node_id in hits:
            name = parts[1] if len(parts) > 1 else ""
            cat  = parts[2] if len(parts) > 2 else ""
            for k, v in targets.items():
                if v == node_id:
                    found[k] = {"id": node_id, "name": name, "category": cat, "line": i}
                    print(f"  ✓ {k}: {node_id} | {name} | {cat} | line {i}")

print("\nRESULTS")
ok = True
for k, v in targets.items():
    if found[k] is None:
        print(f"  ✗ MISSING: {k} ({v})")
        ok = False
    else:
        info = found[k]
        print(f"  ✓ FOUND: {k} -> {info['name']} [{info['category']}] ({info['id']}) at line {info['line']:,}")

if ok:
    print("\nSUCCESS: All critical entities are present.")
    sys.exit(0)
else:
    print("\nFAILURE: One or more critical entities are missing.")
    sys.exit(1)
PY

echo "================================"
echo "End: $(date)"