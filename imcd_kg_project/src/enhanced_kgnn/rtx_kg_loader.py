#!/usr/bin/env python3
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
    kg_path = Path("../../data/kgml_data/bkg_rtxkg2c_v2.7.3")
    
    loader = RTXKGLoader(kg_path)
    
    # Search for our target entities
    search_results = loader.search_entities([
        'adalimumab',
        'castleman',
        'TNF'
    ])
    
    # Get specific entity IDs
    adalimumab_id = "CHEMBL.COMPOUND:CHEMBL1201580"  # Found in search
    castleman_id = "MONDO:0015564"  # Found in search
    
    print(f"\n=== SEARCHING RELATIONSHIPS FOR KEY ENTITIES ===")
    
    # Find relationships for these specific entities
    relationships = loader.find_entity_relationships([
        adalimumab_id,
        castleman_id,
        "TNF"  # Will match any TNF-containing entity
    ])
    
    for entity_id, edges in relationships.items():
        print(f"\n{entity_id} relationships:")
        for edge in edges[:10]:  # Show first 10
            print(f"  {edge['subject']} --{edge['predicate']}--> {edge['object']}")