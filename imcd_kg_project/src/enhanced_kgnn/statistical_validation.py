#!/usr/bin/env python3
"""
Statistical Validation of Experimental Enhancement
Runs multiple seeds to test reproducibility and significance

Following Google's principle: "Test every output like it's trying to kill you"
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List
import json
from datetime import datetime

from rtx_kg_loader import RTXKGLoader
from enhanced_predictor import ExperimentalGraphPredictor, GraphSAGEModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalValidator:
    """Run multiple trials with different seeds to validate results"""
    
    def __init__(self, num_trials=10):
        self.num_trials = num_trials
        self.kg_path = Path("../../data/kgml_data/bkg_rtxkg2c_v2.7.3")
        self.results_dir = Path("../../results/statistical_validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_ranks = []
        self.enhanced_ranks = []
        self.baseline_scores = []
        self.enhanced_scores = []
        self.trial_details = []
        
    def run_single_trial(self, seed: int) -> Dict:
        """Run one trial with given random seed"""
        logger.info(f"="*60)
        logger.info(f"TRIAL {seed + 1}/{self.num_trials} (seed={seed})")
        logger.info(f"="*60)
        
        # Set all random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Initialize predictor
        loader = RTXKGLoader(self.kg_path)
        predictor = ExperimentalGraphPredictor(loader)
        
        # Run baseline model
        logger.info("Training baseline model...")
        baseline_data = predictor.build_graph_with_experimental_features(use_experimental=False)
        baseline_model = GraphSAGEModel(input_dim=3, hidden_dim=64, output_dim=32)
        baseline_model, baseline_data = predictor.train_model(baseline_data, baseline_model, epochs=200)
        
        # Run enhanced model
        logger.info("Training enhanced model...")
        enhanced_data = predictor.build_graph_with_experimental_features(use_experimental=True)
        enhanced_model = GraphSAGEModel(input_dim=4, hidden_dim=64, output_dim=32)
        enhanced_model, enhanced_data = predictor.train_model(enhanced_data, enhanced_model, epochs=200)
        
        # Evaluate rankings
        baseline_ranking = predictor.evaluate_drug_ranking(
            baseline_model, baseline_data, predictor.castleman_id
        )
        enhanced_ranking = predictor.evaluate_drug_ranking(
            enhanced_model, enhanced_data, predictor.castleman_id
        )
        
        # Find adalimumab rankings
        baseline_rank = None
        enhanced_rank = None
        baseline_score = 0
        enhanced_score = 0
        
        for rank, (drug, score) in enumerate(baseline_ranking.items(), 1):
            if drug == predictor.adalimumab_id:
                baseline_rank = rank
                baseline_score = score
                break
        
        for rank, (drug, score) in enumerate(enhanced_ranking.items(), 1):
            if drug == predictor.adalimumab_id:
                enhanced_rank = rank
                enhanced_score = score
                break
        
        result = {
            'seed': seed,
            'baseline_rank': baseline_rank,
            'enhanced_rank': enhanced_rank,
            'baseline_score': baseline_score,
            'enhanced_score': enhanced_score,
            'improvement': baseline_rank - enhanced_rank if baseline_rank and enhanced_rank else None
        }
        
        logger.info(f"Results: Baseline #{baseline_rank} ({baseline_score:.4f}) → Enhanced #{enhanced_rank} ({enhanced_score:.4f})")
        
        return result
    
    def run_all_trials(self):
        """Run all trials and collect results"""
        logger.info(f"\n{'='*60}")
        logger.info(f"STATISTICAL VALIDATION: {self.num_trials} TRIALS")
        logger.info(f"{'='*60}\n")
        
        for seed in range(self.num_trials):
            try:
                result = self.run_single_trial(seed)
                
                self.baseline_ranks.append(result['baseline_rank'])
                self.enhanced_ranks.append(result['enhanced_rank'])
                self.baseline_scores.append(result['baseline_score'])
                self.enhanced_scores.append(result['enhanced_score'])
                self.trial_details.append(result)
                
            except Exception as e:
                logger.error(f"Trial {seed} failed: {e}")
                continue
        
        # Compute statistics
        self.compute_statistics()
        
    def compute_statistics(self):
        """Compute statistical metrics"""
        logger.info(f"\n{'='*60}")
        logger.info("STATISTICAL ANALYSIS")
        logger.info(f"{'='*60}\n")
        
        # Convert to numpy arrays
        baseline_ranks = np.array(self.baseline_ranks)
        enhanced_ranks = np.array(self.enhanced_ranks)
        baseline_scores = np.array(self.baseline_scores)
        enhanced_scores = np.array(self.enhanced_scores)
        improvements = baseline_ranks - enhanced_ranks
        
        # Compute statistics
        stats = {
            'num_trials': len(self.baseline_ranks),
            'baseline': {
                'rank_mean': float(np.mean(baseline_ranks)),
                'rank_std': float(np.std(baseline_ranks)),
                'rank_min': int(np.min(baseline_ranks)),
                'rank_max': int(np.max(baseline_ranks)),
                'score_mean': float(np.mean(baseline_scores)),
                'score_std': float(np.std(baseline_scores))
            },
            'enhanced': {
                'rank_mean': float(np.mean(enhanced_ranks)),
                'rank_std': float(np.std(enhanced_ranks)),
                'rank_min': int(np.min(enhanced_ranks)),
                'rank_max': int(np.max(enhanced_ranks)),
                'score_mean': float(np.mean(enhanced_scores)),
                'score_std': float(np.std(enhanced_scores)),
                'rank_1_count': int(np.sum(enhanced_ranks == 1)),
                'rank_1_percentage': float(100 * np.sum(enhanced_ranks == 1) / len(enhanced_ranks))
            },
            'improvement': {
                'mean': float(np.mean(improvements)),
                'std': float(np.std(improvements)),
                'min': int(np.min(improvements)),
                'max': int(np.max(improvements))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Paired t-test
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_rel(baseline_ranks, enhanced_ranks)
        stats['t_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
        
        # Print results
        logger.info("BASELINE MODEL:")
        logger.info(f"  Rank: {stats['baseline']['rank_mean']:.1f} ± {stats['baseline']['rank_std']:.1f}")
        logger.info(f"  Range: [{stats['baseline']['rank_min']} - {stats['baseline']['rank_max']}]")
        logger.info(f"  Score: {stats['baseline']['score_mean']:.4f} ± {stats['baseline']['score_std']:.4f}")
        
        logger.info("\nENHANCED MODEL:")
        logger.info(f"  Rank: {stats['enhanced']['rank_mean']:.1f} ± {stats['enhanced']['rank_std']:.1f}")
        logger.info(f"  Range: [{stats['enhanced']['rank_min']} - {stats['enhanced']['rank_max']}]")
        logger.info(f"  Score: {stats['enhanced']['score_mean']:.4f} ± {stats['enhanced']['score_std']:.4f}")
        logger.info(f"  Ranked #1: {stats['enhanced']['rank_1_count']}/{len(enhanced_ranks)} trials ({stats['enhanced']['rank_1_percentage']:.1f}%)")
        
        logger.info("\nIMPROVEMENT:")
        logger.info(f"  Mean: {stats['improvement']['mean']:.1f} ± {stats['improvement']['std']:.1f} positions")
        logger.info(f"  Range: [{stats['improvement']['min']} - {stats['improvement']['max']}] positions")
        
        logger.info("\nSTATISTICAL SIGNIFICANCE:")
        logger.info(f"  t-statistic: {stats['t_test']['t_statistic']:.4f}")
        logger.info(f"  p-value: {stats['t_test']['p_value']:.6f}")
        logger.info(f"  Significant (p < 0.05): {'YES ✅' if stats['t_test']['significant'] else 'NO ❌'}")
        
        # Save results
        self.save_results(stats)
        
        return stats
    
    def save_results(self, stats: Dict):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary statistics
        stats_file = self.results_dir / f"statistics_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"\n✅ Statistics saved to: {stats_file}")
        
        # Save detailed trial results
        df = pd.DataFrame(self.trial_details)
        csv_file = self.results_dir / f"trial_details_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"✅ Trial details saved to: {csv_file}")
        
        # Create visualization-ready summary
        summary = {
            'baseline_ranks': self.baseline_ranks,
            'enhanced_ranks': self.enhanced_ranks,
            'baseline_scores': self.baseline_scores,
            'enhanced_scores': self.enhanced_scores
        }
        summary_file = self.results_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Summary saved to: {summary_file}")

def main():
    """Main execution"""
    print("🧪 STATISTICAL VALIDATION")
    print("Testing reproducibility across multiple random seeds")
    print("="*60)
    
    # Check if scipy is available for t-test
    try:
        import scipy
        logger.info("✅ scipy available for statistical tests")
    except ImportError:
        logger.warning("⚠️  scipy not installed - installing now...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    
    # Run validation
    validator = StatisticalValidator(num_trials=10)
    validator.run_all_trials()
    
    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    exit(main())