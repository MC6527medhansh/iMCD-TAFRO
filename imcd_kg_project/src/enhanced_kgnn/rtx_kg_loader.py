"""
RTX-KG2 Knowledge Graph Loader
Architecture: Stream-process large files, filter relevant subgraph
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Set, Tuple, List
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RTXKGLoader:
    """
    Loads RTX-KG2 knowledge graph with focus on drug-disease relationships
    
    Architecture decisions:
    - Stream processing for memory efficiency
    - Filter to relevant entity types early
    - Build NetworkX graph for flexibility
    """
    
    def __init__(self, kg_data_path: Path):
        self.kg_path = kg_data_path
        self.nodes_file = kg_data_path / "tsv_files" / "nodes_c.tsv"
        self.edges_file = kg_data_path / "tsv_files" / "edges_c.tsv"
        
        # Entity types of interest
        self.target_node_types = {
            'biolink:Drug', 'biolink:ChemicalSubstance', 'biolink:SmallMolecule',
            'biolink:Disease', 'biolink:DiseaseOrPhenotypicFeature',
            'biolink:Gene', 'biolink:Protein'
        }
        
        self._validate_files()
    
    def _validate_files(self):
        """Validate required files exist"""
        if not self.nodes_file.exists():
            raise FileNotFoundError(f"Nodes file not found: {self.nodes_file}")
        if not self.edges_file.exists():
            raise FileNotFoundError(f"Edges file not found: {self.edges_file}")
        
        logger.info(f"Nodes file: {self.nodes_file} ({self.nodes_file.stat().st_size / 1e9:.1f}GB)")
        logger.info(f"Edges file: {self.edges_file} ({self.edges_file.stat().st_size / 1e9:.1f}GB)")

    def search_entities(self, search_terms: List[str], max_results: int = 10) -> Dict:
        """
        Search for specific entities in the knowledge graph
        Architecture: Stream processing to find entities without loading full graph
        """
        results = {term: [] for term in search_terms}
        
        logger.info(f"Searching for entities: {search_terms}")
        
        # Search nodes
        with open(self.nodes_file, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num == 0:  # Skip header if present
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                    
                node_id, name, category = parts[0], parts[1], parts[2]
                
                # Search in both ID and name
                search_text = f"{node_id} {name}".lower()
                
                for term in search_terms:
                    if term.lower() in search_text and len(results[term]) < max_results:
                        results[term].append({
                            'id': node_id,
                            'name': name,
                            'category': category,
                            'line': line_num
                        })
        
        return results

    def analyze_relationship_types(self, sample_size: int = 10000) -> Dict:
        """
        Analyze what relationship types exist in the knowledge graph
        Focus on drug-disease connections
        """
        relationship_counts = {}
        
        logger.info(f"Analyzing relationship types from {sample_size} edges...")
        
        with open(self.edges_file, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num >= sample_size:
                    break
                if line_num == 0:  # Skip header
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                    
                subject, object_id, predicate = parts[0], parts[1], parts[2]
                
                relationship_counts[predicate] = relationship_counts.get(predicate, 0) + 1
        
        return dict(sorted(relationship_counts.items(), key=lambda x: x[1], reverse=True))
    
    def find_entity_relationships(self, entity_ids: List[str], max_edges: int = 50) -> Dict:
        """
        Find all relationships involving specific entities
        Critical: Look for actual functional relationships, not just mappings
        """
        entity_relationships = {entity_id: [] for entity_id in entity_ids}
        
        logger.info(f"Finding relationships for entities: {entity_ids}")
        
        with open(self.edges_file, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num == 0:  # Skip header
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                    
                subject, object_id, predicate = parts[0], parts[1], parts[2]
                
                # Check if either subject or object is one of our target entities
                for entity_id in entity_ids:
                    if entity_id in subject or entity_id in object_id:
                        entity_key = entity_id
                        if len(entity_relationships[entity_key]) < max_edges:
                            entity_relationships[entity_key].append({
                                'subject': subject,
                                'object': object_id,
                                'predicate': predicate,
                                'line': line_num
                            })
        
        return entity_relationships
    
    def load_full_graph(self, entity_types_filter=None):
        """
        Load full RTX-KG2 graph into NetworkX
        
        Args:
            entity_types_filter: List of prefixes to filter (e.g., ['CHEMBL.COMPOUND', 'MONDO'])
                                If None, loads all entities
        
        Returns:
            NetworkX DiGraph with nodes and edges
        """
        import networkx as nx
        
        logger.info("Loading full RTX-KG2 graph...")
        G = nx.DiGraph()
        
        # Load nodes
        logger.info(f"Loading nodes from {self.nodes_file}...")
        node_count = 0
        with open(self.nodes_file, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num == 0:  # Skip header
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                node_id, name, category = parts[0], parts[1], parts[2]
                
                # Filter by entity type if specified
                if entity_types_filter:
                    if not any(node_id.startswith(prefix) for prefix in entity_types_filter):
                        continue
                
                G.add_node(node_id, name=name, category=category)
                node_count += 1
                
                if node_count % 100000 == 0:
                    logger.info(f"  Loaded {node_count:,} nodes...")
        
        logger.info(f"✓ Loaded {node_count:,} nodes")
        
        # Load edges
        logger.info(f"Loading edges from {self.edges_file}...")
        edge_count = 0
        with open(self.edges_file, 'r') as f:
            for line_num, line in enumerate(f):
                if line_num == 0:  # Skip header
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                subject, obj, predicate = parts[0], parts[1], parts[2]
                
                # Only add edge if both nodes exist in graph
                if G.has_node(subject) and G.has_node(obj):
                    G.add_edge(subject, obj, predicate=predicate)
                    edge_count += 1
                    
                    if edge_count % 100000 == 0:
                        logger.info(f"  Loaded {edge_count:,} edges...")
        
        logger.info(f"✓ Loaded {edge_count:,} edges")
        logger.info(f"✓ Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        
        return G
    
    def validate_critical_entities(self, graph, adalimumab_id, castleman_id, tnf_id):
        """
        Validate that critical entities and paths exist in loaded graph
        
        Returns:
            Dict with validation results
        """
        import networkx as nx
        
        logger.info("Validating critical entities and paths...")
        
        results = {
            'adalimumab_exists': graph.has_node(adalimumab_id),
            'castleman_exists': graph.has_node(castleman_id),
            'tnf_exists': graph.has_node(tnf_id),
            'ada_tnf_path': False,
            'tnf_castleman_path': False,
            'complete_pathway': False
        }
        
        # Check direct connections
        results['ada_tnf_direct'] = graph.has_edge(adalimumab_id, tnf_id) or \
                                     graph.has_edge(tnf_id, adalimumab_id)
        results['tnf_castleman_direct'] = graph.has_edge(tnf_id, castleman_id) or \
                                           graph.has_edge(castleman_id, tnf_id)
        
        # Check paths (up to 3 hops)
        if results['adalimumab_exists'] and results['tnf_exists']:
            try:
                path = nx.shortest_path(graph.to_undirected(), adalimumab_id, tnf_id)
                results['ada_tnf_path'] = len(path) <= 4  # 3 hops max
                results['ada_tnf_path_length'] = len(path) - 1
                logger.info(f"  Adalimumab → TNF: {len(path)-1} hops")
            except nx.NetworkXNoPath:
                logger.warning("  No path found: Adalimumab → TNF")
        
        if results['tnf_exists'] and results['castleman_exists']:
            try:
                path = nx.shortest_path(graph.to_undirected(), tnf_id, castleman_id)
                results['tnf_castleman_path'] = len(path) <= 4  # 3 hops max
                results['tnf_castleman_path_length'] = len(path) - 1
                logger.info(f"  TNF → Castleman: {len(path)-1} hops")
            except nx.NetworkXNoPath:
                logger.warning("  No path found: TNF → Castleman")
        
        # Complete pathway check
        results['complete_pathway'] = results['ada_tnf_path'] and results['tnf_castleman_path']
        
        return results
    
    def save_graph(self, graph, output_path):
        """Save graph to pickle file"""
        import pickle
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving graph to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        size_gb = output_path.stat().st_size / 1e9
        logger.info(f"✓ Graph saved ({size_gb:.2f} GB)")
        
    def load_saved_graph(self, graph_path):
        """Load pre-computed graph from pickle"""
        import pickle
        logger.info(f"Loading saved graph from {graph_path}...")
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        logger.info(f"✓ Loaded graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")
        return graph


def inspect_data_structure(kg_data_path: Path):
    """
    Inspect the actual structure of RTX-KG2 files
    Critical first step before building loader
    """
    nodes_file = kg_data_path / "tsv_files" / "nodes_c.tsv"
    edges_file = kg_data_path / "tsv_files" / "edges_c.tsv"
    
    logger.info("Inspecting RTX-KG2 data structure...")
    
    # Read first few lines to understand format
    logger.info("Node file structure:")
    with open(nodes_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:  # First 5 lines
                print(f"Line {i}: {line.strip()}")
            else:
                break
    
    logger.info("\nEdge file structure:")
    with open(edges_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:  # First 5 lines
                print(f"Line {i}: {line.strip()}")
            else:
                break
            



if __name__ == "__main__":
    """Phase 1.1: Entity Verification"""
    import json
    
    kg_path = Path("../../data/kgml_data/bkg_rtxkg2c_v2.7.3")
    loader = RTXKGLoader(kg_path)
    
    # Critical entities from PROJECT_CONTEXT.md
    TARGET_ENTITIES = {
        'adalimumab': 'CHEMBL.COMPOUND:CHEMBL1201580',
        'castleman': 'MONDO:0015564',
        'tnf_gene': 'UniProtKB:P01375'
    }
    
    print("="*60)
    print("PHASE 1.1: ENTITY VERIFICATION")
    print("="*60)
    
    # Search for entities
    search_terms = list(TARGET_ENTITIES.values())
    results = loader.search_entities(search_terms, max_results=5)
    
    verification = {}
    all_found = True
    
    for name, entity_id in TARGET_ENTITIES.items():
        found = len(results[entity_id]) > 0
        
        if found:
            entity = results[entity_id][0]
            print(f"\n✅ {name.upper()}: FOUND")
            print(f"   ID: {entity['id']}")
            print(f"   Name: {entity['name']}")
            print(f"   Category: {entity['category']}")
            verification[name] = {
                'found': True,
                'id': entity['id'],
                'name': entity['name'],
                'category': entity['category']
            }
        else:
            print(f"\n❌ {name.upper()}: NOT FOUND")
            print(f"   Expected: {entity_id}")
            verification[name] = {'found': False, 'expected': entity_id}
            all_found = False
    
    # Save results
    results_dir = Path("../../results/phase_1_1")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "entity_verification.json", 'w') as f:
        json.dump(verification, f, indent=2)
    
    print("\n" + "="*60)
    if all_found:
        print("✅ PHASE 1.1 COMPLETE - All entities verified")
    else:
        print("❌ PHASE 1.1 FAILED - Missing entities")
        exit(1)