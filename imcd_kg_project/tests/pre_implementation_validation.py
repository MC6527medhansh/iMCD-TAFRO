#!/usr/bin/env python3
"""
Pre-Implementation Validation Suite
Location: tests/pre_implementation_validation.py

PURPOSE: Verify EVERY assumption before writing mechanism_graph_loader.py
Following Google's principle: "Test every output like it's trying to kill you"

Run this FIRST. If all pass → proceed to Phase 1.
If any fail → stop and debug.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "enhanced_kgnn"))

from rtx_kg_loader import RTXKGLoader

class PreImplementationValidator:
    """
    Validate critical assumptions before building mechanism graph
    
    Each test answers a specific question:
    1. Do we have the data?
    2. Do critical entities exist?
    3. Do mechanism paths exist?
    4. Can we handle the memory?
    5. What's the graph topology?
    """
    
    def __init__(self):
        self.kg_path = PROJECT_ROOT / "data" / "kgml_data" / "bkg_rtxkg2c_v2.7.3"
        self.results = {}
        
        # Critical entity IDs
        self.adalimumab_id = "CHEMBL.COMPOUND:CHEMBL1201580"
        self.castleman_id = "MONDO:0015564"
        self.tnf_ids = [
            "UniProtKB:P01375",      # TNF protein
            "NCBIGene:7124",         # TNF gene
            "HGNC:11892"             # HGNC TNF
        ]
    
    def test_1_data_files_exist(self):
        """TEST 1: Do we have the full RTX-KG2 files?"""
        logger.info("="*60)
        logger.info("TEST 1: Data Files Existence")
        logger.info("="*60)
        
        nodes_file = self.kg_path / "tsv_files" / "nodes_c.tsv"
        edges_file = self.kg_path / "tsv_files" / "edges_c.tsv"
        
        # Check existence
        nodes_exist = nodes_file.exists()
        edges_exist = edges_file.exists()
        
        logger.info(f"  Nodes file: {'✅ FOUND' if nodes_exist else '❌ MISSING'}")
        logger.info(f"  Path: {nodes_file}")
        
        logger.info(f"  Edges file: {'✅ FOUND' if edges_exist else '❌ MISSING'}")
        logger.info(f"  Path: {edges_file}")
        
        if nodes_exist and edges_exist:
            # Get sizes
            nodes_size_gb = nodes_file.stat().st_size / 1e9
            edges_size_gb = edges_file.stat().st_size / 1e9
            
            logger.info(f"  Nodes size: {nodes_size_gb:.2f} GB")
            logger.info(f"  Edges size: {edges_size_gb:.2f} GB")
            logger.info(f"  Total: {nodes_size_gb + edges_size_gb:.2f} GB")
            
            self.results['test_1'] = {
                'status': 'PASS',
                'nodes_size_gb': nodes_size_gb,
                'edges_size_gb': edges_size_gb
            }
            return True
        else:
            logger.error("  ❌ FAIL: Missing data files!")
            logger.error("  Download from: https://zenodo.org/record/7582233")
            self.results['test_1'] = {'status': 'FAIL', 'reason': 'files_missing'}
            return False
    
    def test_2_memory_capacity(self):
        """TEST 2: Can we load the graph into memory?"""
        logger.info("="*60)
        logger.info("TEST 2: Memory Capacity Check")
        logger.info("="*60)
        
        try:
            import psutil
            
            # Get available memory
            available_ram = psutil.virtual_memory().available / 1e9
            total_ram = psutil.virtual_memory().total / 1e9
            
            logger.info(f"  Total RAM: {total_ram:.1f} GB")
            logger.info(f"  Available RAM: {available_ram:.1f} GB")
            
            # Estimate needed memory (conservative: 4x file size)
            if 'test_1' in self.results and self.results['test_1']['status'] == 'PASS':
                file_size = self.results['test_1']['nodes_size_gb'] + self.results['test_1']['edges_size_gb']
                estimated_memory = file_size * 4
                
                logger.info(f"  Estimated memory needed: {estimated_memory:.1f} GB")
                
                if available_ram > estimated_memory:
                    logger.info(f"  ✅ PASS: Sufficient memory ({available_ram:.1f}GB > {estimated_memory:.1f}GB)")
                    self.results['test_2'] = {
                        'status': 'PASS',
                        'available_gb': available_ram,
                        'estimated_needed_gb': estimated_memory
                    }
                    return True
                else:
                    logger.warning(f"  ⚠️  WARNING: Tight memory")
                    logger.warning(f"  Available: {available_ram:.1f}GB, Estimated need: {estimated_memory:.1f}GB")
                    logger.warning(f"  Recommendation: Use streaming or sample subgraph")
                    self.results['test_2'] = {
                        'status': 'WARNING',
                        'available_gb': available_ram,
                        'estimated_needed_gb': estimated_memory
                    }
                    return True
            else:
                logger.error("  ❌ SKIP: Test 1 must pass first")
                return False
                
        except ImportError:
            logger.error("  ❌ FAIL: psutil not installed")
            logger.error("  Install: pip install psutil")
            self.results['test_2'] = {'status': 'FAIL', 'reason': 'psutil_missing'}
            return False
    
    def test_3_tnf_entity_exists(self):
        """TEST 3: Does TNF gene/protein exist in RTX-KG2?"""
        logger.info("="*60)
        logger.info("TEST 3: TNF Entity Existence")
        logger.info("="*60)
        
        try:
            loader = RTXKGLoader(self.kg_path)
            
            # Search for TNF
            logger.info("  Searching for TNF entities...")
            search_results = loader.search_entities(['TNF', 'tumor necrosis factor'], max_results=20)
            
            tnf_found = []
            for term, results in search_results.items():
                logger.info(f"  Search term '{term}': {len(results)} results")
                for result in results[:5]:
                    logger.info(f"    - {result['id']}: {result['name']}")
                    if any(tnf_id in result['id'] for tnf_id in self.tnf_ids):
                        tnf_found.append(result['id'])
            
            if len(tnf_found) > 0:
                logger.info(f"  ✅ PASS: Found {len(tnf_found)} TNF entities")
                logger.info(f"  Primary TNF IDs: {tnf_found}")
                self.results['test_3'] = {
                    'status': 'PASS',
                    'tnf_entities': tnf_found
                }
                return True
            else:
                logger.error("  ❌ FAIL: No TNF entities found!")
                logger.error("  Expected one of: " + ', '.join(self.tnf_ids))
                self.results['test_3'] = {'status': 'FAIL', 'reason': 'tnf_not_found'}
                return False
                
        except Exception as e:
            logger.error(f"  ❌ FAIL: {e}")
            self.results['test_3'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_4_tnf_has_edges(self):
        """TEST 4: Does TNF have edges to drugs/diseases?"""
        logger.info("="*60)
        logger.info("TEST 4: TNF Connectivity Analysis")
        logger.info("="*60)
        
        try:
            loader = RTXKGLoader(self.kg_path)
            
            # Check each TNF ID
            for tnf_id in self.tnf_ids:
                logger.info(f"\n  Checking {tnf_id}...")
                
                relationships = loader.find_entity_relationships([tnf_id], max_edges=100)
                
                if tnf_id in relationships:
                    edges = relationships[tnf_id]
                    logger.info(f"    Found {len(edges)} edges")
                    
                    if len(edges) > 0:
                        # Analyze edge types
                        predicates = {}
                        connected_to_drugs = 0
                        connected_to_diseases = 0
                        
                        for edge in edges:
                            pred = edge['predicate']
                            predicates[pred] = predicates.get(pred, 0) + 1
                            
                            if 'CHEMBL' in edge['subject'] or 'CHEMBL' in edge['object']:
                                connected_to_drugs += 1
                            if 'MONDO' in edge['subject'] or 'MONDO' in edge['object']:
                                connected_to_diseases += 1
                        
                        logger.info(f"    Connected to drugs: {connected_to_drugs}")
                        logger.info(f"    Connected to diseases: {connected_to_diseases}")
                        logger.info(f"    Relationship types:")
                        for pred, count in sorted(predicates.items(), key=lambda x: x[1], reverse=True)[:5]:
                            logger.info(f"      - {pred}: {count}")
                        
                        if len(edges) >= 10:
                            logger.info(f"  ✅ PASS: {tnf_id} has {len(edges)} edges")
                            self.results['test_4'] = {
                                'status': 'PASS',
                                'tnf_id': tnf_id,
                                'total_edges': len(edges),
                                'drug_edges': connected_to_drugs,
                                'disease_edges': connected_to_diseases,
                                'predicates': predicates
                            }
                            return True
                else:
                    logger.warning(f"    No relationships found for {tnf_id}")
            
            logger.error("  ❌ FAIL: No TNF entities have sufficient edges")
            self.results['test_4'] = {'status': 'FAIL', 'reason': 'insufficient_edges'}
            return False
            
        except Exception as e:
            logger.error(f"  ❌ FAIL: {e}")
            self.results['test_4'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_5_adalimumab_connectivity(self):
        """TEST 5: Does adalimumab connect to anything TNF-related?"""
        logger.info("="*60)
        logger.info("TEST 5: Adalimumab-TNF Connection")
        logger.info("="*60)
        
        try:
            loader = RTXKGLoader(self.kg_path)
            
            logger.info(f"  Searching for {self.adalimumab_id} relationships...")
            relationships = loader.find_entity_relationships([self.adalimumab_id], max_edges=100)
            
            if self.adalimumab_id in relationships:
                edges = relationships[self.adalimumab_id]
                logger.info(f"    Found {len(edges)} edges")
                
                # Look for TNF connections
                tnf_connections = []
                for edge in edges:
                    edge_str = f"{edge['subject']} --{edge['predicate']}--> {edge['object']}"
                    if any(tnf_id in edge_str for tnf_id in self.tnf_ids) or 'TNF' in edge_str.upper():
                        tnf_connections.append(edge)
                        logger.info(f"    ⭐ TNF connection: {edge_str}")
                
                # Show sample edges even if no TNF
                logger.info(f"\n    Sample edges:")
                for edge in edges[:10]:
                    logger.info(f"      {edge['subject'][:40]}... --{edge['predicate']}--> {edge['object'][:40]}...")
                
                if len(tnf_connections) > 0:
                    logger.info(f"  ✅ PASS: Found {len(tnf_connections)} adalimumab-TNF connections")
                    self.results['test_5'] = {
                        'status': 'PASS',
                        'direct_tnf_connections': len(tnf_connections),
                        'total_edges': len(edges)
                    }
                    return True
                else:
                    logger.warning(f"  ⚠️  WARNING: No direct TNF connections found")
                    logger.warning(f"  May need multi-hop path or different TNF ID")
                    self.results['test_5'] = {
                        'status': 'WARNING',
                        'direct_tnf_connections': 0,
                        'total_edges': len(edges)
                    }
                    return True
            else:
                logger.error(f"  ❌ FAIL: No edges found for adalimumab")
                self.results['test_5'] = {'status': 'FAIL', 'reason': 'no_edges'}
                return False
                
        except Exception as e:
            logger.error(f"  ❌ FAIL: {e}")
            self.results['test_5'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_6_castleman_connectivity(self):
        """TEST 6: Does Castleman disease connect to TNF?"""
        logger.info("="*60)
        logger.info("TEST 6: Castleman-TNF Connection")
        logger.info("="*60)
        
        try:
            loader = RTXKGLoader(self.kg_path)
            
            logger.info(f"  Searching for {self.castleman_id} relationships...")
            relationships = loader.find_entity_relationships([self.castleman_id], max_edges=100)
            
            if self.castleman_id in relationships:
                edges = relationships[self.castleman_id]
                logger.info(f"    Found {len(edges)} edges")
                
                # Look for TNF or gene connections
                tnf_connections = []
                gene_connections = 0
                
                for edge in edges:
                    edge_str = f"{edge['subject']} --{edge['predicate']}--> {edge['object']}"
                    if any(tnf_id in edge_str for tnf_id in self.tnf_ids) or 'TNF' in edge_str.upper():
                        tnf_connections.append(edge)
                        logger.info(f"    ⭐ TNF connection: {edge_str}")
                    
                    if 'UniProtKB' in edge_str or 'NCBIGene' in edge_str or 'HGNC' in edge_str:
                        gene_connections += 1
                
                logger.info(f"    Connections to genes/proteins: {gene_connections}")
                logger.info(f"\n    Sample edges:")
                for edge in edges[:10]:
                    logger.info(f"      {edge['subject'][:40]}... --{edge['predicate']}--> {edge['object'][:40]}...")
                
                if len(tnf_connections) > 0 or gene_connections > 0:
                    logger.info(f"  ✅ PASS: Castleman has gene/TNF connections")
                    self.results['test_6'] = {
                        'status': 'PASS',
                        'tnf_connections': len(tnf_connections),
                        'gene_connections': gene_connections,
                        'total_edges': len(edges)
                    }
                    return True
                else:
                    logger.warning(f"  ⚠️  WARNING: No gene connections found")
                    logger.warning(f"  Castleman may be isolated in graph")
                    self.results['test_6'] = {
                        'status': 'WARNING',
                        'total_edges': len(edges)
                    }
                    return True
            else:
                logger.error(f"  ❌ FAIL: No edges found for Castleman")
                self.results['test_6'] = {'status': 'FAIL', 'reason': 'no_edges'}
                return False
                
        except Exception as e:
            logger.error(f"  ❌ FAIL: {e}")
            self.results['test_6'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def test_7_graph_topology_sampling(self):
        """TEST 7: What's the actual graph structure?"""
        logger.info("="*60)
        logger.info("TEST 7: Graph Topology Sampling")
        logger.info("="*60)
        
        try:
            logger.info("  Sampling 10,000 nodes from RTX-KG2...")
            
            nodes_file = self.kg_path / "tsv_files" / "nodes_c.tsv"
            
            # Count entity types in sample
            entity_types = {}
            sample_size = 10000
            
            with open(nodes_file, 'r') as f:
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    if i == 0:  # Skip header
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        node_id = parts[0]
                        category = parts[2]
                        
                        # Extract prefix
                        prefix = node_id.split(':')[0] if ':' in node_id else 'unknown'
                        entity_types[prefix] = entity_types.get(prefix, 0) + 1
            
            logger.info(f"  Entity type distribution (sample of {sample_size}):")
            for prefix, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:15]:
                logger.info(f"    {prefix}: {count} ({100*count/sample_size:.1f}%)")
            
            # Check for key entity types
            has_drugs = any(p in entity_types for p in ['CHEMBL.COMPOUND', 'DRUGBANK'])
            has_diseases = 'MONDO' in entity_types or 'DOID' in entity_types
            has_genes = any(p in entity_types for p in ['UniProtKB', 'NCBIGene', 'HGNC'])
            
            logger.info(f"\n  Key entity types present:")
            logger.info(f"    Drugs: {'✅' if has_drugs else '❌'}")
            logger.info(f"    Diseases: {'✅' if has_diseases else '❌'}")
            logger.info(f"    Genes/Proteins: {'✅' if has_genes else '❌'}")
            
            if has_drugs and has_diseases and has_genes:
                logger.info(f"  ✅ PASS: Graph has drugs, diseases, and genes")
                self.results['test_7'] = {
                    'status': 'PASS',
                    'entity_types': entity_types,
                    'has_drugs': has_drugs,
                    'has_diseases': has_diseases,
                    'has_genes': has_genes
                }
                return True
            else:
                logger.error(f"  ❌ FAIL: Missing critical entity types")
                self.results['test_7'] = {
                    'status': 'FAIL',
                    'entity_types': entity_types
                }
                return False
                
        except Exception as e:
            logger.error(f"  ❌ FAIL: {e}")
            self.results['test_7'] = {'status': 'FAIL', 'error': str(e)}
            return False
    
    def run_all_tests(self):
        """Run all validation tests"""
        logger.info("\n" + "="*60)
        logger.info("PRE-IMPLEMENTATION VALIDATION SUITE")
        logger.info("Verifying assumptions before building mechanism graph")
        logger.info("="*60 + "\n")
        
        tests = [
            self.test_1_data_files_exist,
            self.test_2_memory_capacity,
            self.test_3_tnf_entity_exists,
            self.test_4_tnf_has_edges,
            self.test_5_adalimumab_connectivity,
            self.test_6_castleman_connectivity,
            self.test_7_graph_topology_sampling
        ]
        
        passed = 0
        failed = 0
        warnings = 0
        
        for test_func in tests:
            try:
                result = test_func()
                if result:
                    if self.results.get(test_func.__name__.replace('test_', 'test_'), {}).get('status') == 'WARNING':
                        warnings += 1
                    else:
                        passed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Test {test_func.__name__} crashed: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
            
            print()  # Spacing
        
        # Summary
        logger.info("="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Passed: {passed}/7")
        logger.info(f"⚠️  Warnings: {warnings}/7")
        logger.info(f"❌ Failed: {failed}/7")
        
        for test_name, result in self.results.items():
            status = result['status']
            symbol = "✅" if status == 'PASS' else "⚠️" if status == 'WARNING' else "❌"
            logger.info(f"{symbol} {test_name}: {status}")
        
        # Save results
        results_file = PROJECT_ROOT / "results" / "pre_implementation_validation.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\n💾 Results saved to: {results_file}")
        
        # Decision
        logger.info("\n" + "="*60)
        logger.info("DECISION GATE")
        logger.info("="*60)
        
        if failed == 0:
            logger.info("✅ ALL CRITICAL TESTS PASSED")
            logger.info("→ PROCEED TO PHASE 1: Mechanism Graph Construction")
            return True
        else:
            logger.error(f"❌ {failed} TESTS FAILED")
            logger.error("→ FIX ISSUES BEFORE PROCEEDING")
            logger.error("\nFailure Analysis:")
            for test_name, result in self.results.items():
                if result['status'] == 'FAIL':
                    logger.error(f"  {test_name}: {result.get('reason', result.get('error', 'unknown'))}")
            return False

def main():
    """Run pre-implementation validation"""
    print("🔬 PRE-IMPLEMENTATION VALIDATION")
    print("Testing assumptions before writing mechanism_graph_loader.py")
    print("="*60 + "\n")
    
    validator = PreImplementationValidator()
    all_passed = validator.run_all_tests()
    
    if all_passed:
        print("\n🎉 VALIDATION COMPLETE - READY TO PROCEED")
        print("Next step: Create mechanism_graph_loader.py")
        return 0
    else:
        print("\n⚠️  VALIDATION FAILED - REVIEW ISSUES ABOVE")
        print("Fix failures before proceeding to Phase 1")
        return 1

if __name__ == "__main__":
    exit(main())