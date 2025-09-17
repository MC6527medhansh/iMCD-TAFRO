"""
Single-cell RNA-seq analysis pipeline for iMCD project
Following Luecken & Theis best practices
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from config import PROCESSED_DATA_DIR, RESULTS_DIR, IMCD_EXPERIMENTAL_DATA
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Scanpy settings
sc.settings.verbosity = 3  # verbosity level
sc.settings.set_figure_params(dpi=80, facecolor='white')

class iMCDSingleCellAnalyzer:
    """
    Single-cell analyzer following Luecken & Theis workflow
    Adapted for iMCD-TAFRO project goals
    """
    
    def __init__(self, save_results=True):
        self.save_results = save_results
        self.results_dir = RESULTS_DIR / "scrna_analysis"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Key genes from iMCD paper
        self.key_genes = ['TNF', 'IL6', 'IL6R', 'IFNG', 'IL5', 'STAT3', 'NFKB1']
        self.tnf_pathway_genes = ['TNF', 'TNFRSF1A', 'TNFRSF1B', 'NFKB1', 'RELA', 'IKBKA']
        
    def load_example_data(self):
        """Load example PBMC data to practice the pipeline"""
        logger.info("Loading example PBMC data...")
        
        # Load reduced PBMC dataset for practice
        adata = sc.datasets.pbmc68k_reduced()
        logger.info(f"Loaded data: {adata.n_obs} cells x {adata.n_vars} genes")
        
        return adata
    
    def quality_control(self, adata):
        """
        Quality control following Luecken & Theis recommendations
        """
        logger.info("Starting quality control...")
        
        # Identify mitochondrial genes
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
        adata.var['hb'] = adata.var_names.str.contains('^HB[^(P)]')
        
        # Calculate basic metrics manually to avoid scanpy issues
        if hasattr(adata.X, 'getnnz'):
            # Sparse matrix
            adata.obs['n_genes_by_counts'] = adata.X.getnnz(axis=1)
            adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        else:
            # Dense matrix
            adata.obs['n_genes_by_counts'] = (adata.X > 0).sum(axis=1)
            adata.obs['total_counts'] = adata.X.sum(axis=1)
        
        # Calculate mitochondrial percentage manually
        if adata.var['mt'].any():
            mt_genes = adata.var['mt']
            if hasattr(adata.X, 'getnnz'):
                mt_counts = np.array(adata.X[:, mt_genes].sum(axis=1)).flatten()
            else:
                mt_counts = adata.X[:, mt_genes].sum(axis=1)
            adata.obs['pct_counts_mt'] = (mt_counts / adata.obs['total_counts']) * 100
        else:
            # No MT genes found, set to 0
            adata.obs['pct_counts_mt'] = 0.0
            logger.warning("No mitochondrial genes found, setting pct_counts_mt to 0")
        
        logger.info("QC metrics calculated")
        return adata

    def filter_cells_genes(self, adata):
        """
        Filter cells and genes based on QC metrics
        """
        logger.info("Filtering cells and genes...")
        
        # Store original counts
        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars
        
        # Filter genes: expressed in at least 3 cells
        sc.pp.filter_genes(adata, min_cells=3)
        
        # Less aggressive cell filtering for small datasets
        min_genes_threshold = min(200, int(adata.n_vars * 0.1))  # Adaptive threshold
        sc.pp.filter_cells(adata, min_genes=min_genes_threshold)
        
        # Only apply MT and gene count filters if we have enough cells
        if adata.n_obs > 100:
            # Filter high mitochondrial content
            if adata.obs['pct_counts_mt'].max() > 0:
                adata = adata[adata.obs['pct_counts_mt'] < 20, :].copy()
            
            # Filter very high gene counts (potential doublets)
            upper_lim = np.quantile(adata.obs['n_genes_by_counts'].values, 0.98)
            adata = adata[adata.obs['n_genes_by_counts'] < upper_lim, :].copy()
        else:
            logger.info("Skipping aggressive filtering - too few cells")
        
        logger.info(f"Filtered: {n_cells_before} -> {adata.n_obs} cells")
        logger.info(f"Filtered: {n_genes_before} -> {adata.n_vars} genes")
        
        return adata
    
    def normalize_and_scale(self, adata):
        """
        Normalization and scaling following best practices
        """
        logger.info("Normalizing and scaling...")
        
        # Save raw counts
        adata.raw = adata
        
        # Normalize to 10,000 reads per cell
        sc.pp.normalize_total(adata, target_sum=1e4)
        
        # Log transform
        sc.pp.log1p(adata)
        
        # Remove any NaN or infinite values
        adata.X[np.isnan(adata.X)] = 0
        adata.X[np.isinf(adata.X)] = 0
        
        # Find highly variable genes - simple approach
        gene_var = np.var(adata.X, axis=0)
        top_genes = np.argsort(gene_var)[-2000:]
        adata.var['highly_variable'] = False
        adata.var.iloc[top_genes, adata.var.columns.get_loc('highly_variable')] = True
        
        # Scale data
        sc.pp.scale(adata, max_value=10)
        
        logger.info("Normalization complete")
        return adata
    
    def dimension_reduction_clustering(self, adata):
        """
        PCA, UMAP, and clustering
        """
        logger.info("Running dimension reduction and clustering...")
        
        # Principal component analysis
        sc.tl.pca(adata, svd_solver='arpack')
        
        # Compute neighborhood graph
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        
        # UMAP embedding
        sc.tl.umap(adata)
        
        # Leiden clustering
        sc.tl.leiden(adata, resolution=0.5)
        
        logger.info("Clustering complete")
        return adata
    
    def analyze_tnf_expression(self, adata):
        """
        Analyze TNF expression specifically (key for iMCD project)
        """
        logger.info("Analyzing TNF expression patterns...")
        
        # Check if TNF is in the dataset
        if 'TNF' not in adata.var_names:
            logger.warning("TNF gene not found in dataset")
            return adata
        
        # Calculate TNF expression per cluster
        tnf_by_cluster = []
        for cluster in adata.obs['leiden'].cat.categories:
            cluster_cells = adata.obs['leiden'] == cluster
            tnf_expr = adata.raw[cluster_cells, 'TNF'].X.mean()
            tnf_by_cluster.append({
                'cluster': cluster,
                'tnf_mean_expression': tnf_expr,
                'n_cells': cluster_cells.sum()
            })
        
        tnf_df = pd.DataFrame(tnf_by_cluster)
        logger.info("TNF expression by cluster:")
        print(tnf_df.to_string(index=False))
        
        # Save TNF analysis
        if self.save_results:
            tnf_df.to_csv(self.results_dir / "tnf_expression_by_cluster.csv", index=False)
        
        return adata
    
    def identify_cd4_tcells(self, adata):
        """
        Identify CD4+ T cells (key cell type from iMCD paper)
        """
        logger.info("Identifying CD4+ T cells...")
        
        # T cell markers
        tcell_markers = ['CD3D', 'CD3E', 'CD3G']
        cd4_markers = ['CD4']
        cd8_markers = ['CD8A', 'CD8B']
        
        # Check which markers are available
        available_tcell = [gene for gene in tcell_markers if gene in adata.var_names]
        available_cd4 = [gene for gene in cd4_markers if gene in adata.var_names]
        available_cd8 = [gene for gene in cd8_markers if gene in adata.var_names]
        
        logger.info(f"Available T cell markers: {available_tcell}")
        logger.info(f"Available CD4 markers: {available_cd4}")
        logger.info(f"Available CD8 markers: {available_cd8}")
        
        if available_tcell and available_cd4:
            # Calculate T cell and CD4 scores
            sc.tl.score_genes(adata, gene_list=available_tcell, score_name='tcell_score')
            sc.tl.score_genes(adata, gene_list=available_cd4, score_name='cd4_score')
            
            if available_cd8:
                sc.tl.score_genes(adata, gene_list=available_cd8, score_name='cd8_score')
                
            # Identify potential CD4+ T cells
            adata.obs['potential_cd4_tcell'] = (
                (adata.obs['tcell_score'] > 0.1) & 
                (adata.obs['cd4_score'] > 0.05)
            )
            
            n_cd4_tcells = adata.obs['potential_cd4_tcell'].sum()
            logger.info(f"Identified {n_cd4_tcells} potential CD4+ T cells")
            
        return adata
    
    def run_full_pipeline(self):
        """
        Run the complete single-cell analysis pipeline
        """
        logger.info("🧬 Starting iMCD single-cell analysis pipeline...")
        
        # Load data
        adata = self.load_example_data()
        
        # QC and filtering
        adata = self.quality_control(adata)
        adata = self.filter_cells_genes(adata)
        
        # Normalization and scaling
        adata = self.normalize_and_scale(adata)
        
        # Dimension reduction and clustering
        adata = self.dimension_reduction_clustering(adata)
        
        # iMCD-specific analyses
        adata = self.analyze_tnf_expression(adata)
        adata = self.identify_cd4_tcells(adata)
        
        # Save processed data
        if self.save_results:
            adata.write(self.results_dir / "processed_pbmc_data.h5ad")
            logger.info(f"Results saved to {self.results_dir}")
        
        logger.info("✅ Single-cell analysis pipeline complete!")
        return adata

def main():
    """Main function to run single-cell analysis"""
    print("🧬 iMCD Single-Cell RNA-seq Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = iMCDSingleCellAnalyzer(save_results=True)
    
    # Run pipeline
    adata = analyzer.run_full_pipeline()
    
    # Print key findings related to iMCD
    print("\n📊 KEY FINDINGS FOR iMCD PROJECT:")
    print("-" * 40)
    
    exp_data = IMCD_EXPERIMENTAL_DATA['scrna_seq']
    print(f"• Paper finding: TNF {exp_data['naive_cd4_tnf_fold_change']} log2 fold higher in naive CD4+ T cells")
    print(f"• Paper samples: {exp_data['samples']['flare']} flare patients vs {exp_data['samples']['healthy']} healthy")
    print("• Our analysis: Practiced the pipeline on example data")
    print("• Next step: Apply this to real iMCD data when available")
    
    return 0

if __name__ == "__main__":
    exit(main())