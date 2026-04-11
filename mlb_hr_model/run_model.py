"""
MLB HR Model — Quickstart Runner
=================================

Run this script to execute the full pipeline end-to-end.
Make sure you have the dependencies installed:

    pip install pybaseball pandas numpy scikit-learn lightgbm

Usage:
    python run_model.py                    # Full pipeline (takes 30+ min for data pull)
    python run_model.py --test             # Quick test with small date range
    python run_model.py --from-cache       # Skip data pull, use cached parquet
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_data_pipeline import run_phase1_pipeline, CONFIG
from phase2_model_training import run_phase2_pipeline


def main():
    parser = argparse.ArgumentParser(description='MLB HR Prop Model')
    parser.add_argument('--test', action='store_true', 
                        help='Run with small date range for testing')
    parser.add_argument('--from-cache', action='store_true',
                        help='Load data from cached parquet file')
    parser.add_argument('--start', type=str, default=None,
                        help='Override start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                        help='Override end date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Determine date range
    if args.test:
        start = '2024-07-01'
        end = '2024-07-31'
        cache_path = 'data/statcast_test_jul2024.parquet'
        train_end = '2024-07-20'
        val_end = '2024-07-27'
        print("\n🧪 TEST MODE — using July 2024 only\n")
    else:
        start = args.start or CONFIG['train_start']
        end = args.end or CONFIG['train_end']
        cache_path = f'data/statcast_{start}_{end}.parquet'
        train_end = '2024-06-30'
        val_end = '2024-09-30'
        print(f"\n⚾ FULL MODE — {start} to {end}\n")
    
    if args.from_cache and not os.path.exists(cache_path):
        print(f"Cache file not found: {cache_path}")
        print("Run without --from-cache first to download data")
        return
    
    # ---- PHASE 1: Data Pipeline ----
    print("=" * 60)
    print("PHASE 1: Data Pipeline & Feature Engineering")
    print("=" * 60)
    
    phase1_result = run_phase1_pipeline(start, end, cache_path)
    
    # ---- PHASE 2: Model Training ----
    print("\n" + "=" * 60)
    print("PHASE 2: Model Training & Calibration")  
    print("=" * 60)
    
    phase2_result = run_phase2_pipeline(
        phase1_result,
        train_end=train_end,
        val_end=val_end
    )
    
    # ---- Summary ----
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: models/")
    print(f"Feature importance: {phase2_result['feature_importance'].head(5)['feature'].tolist()}")
    print(f"\nCalibration ECE: {phase2_result['eval_results']['ece']:.4f}")
    print(f"AUC-ROC: {phase2_result['eval_results']['auc_roc']:.4f}")
    
    if phase2_result['game_probs'] is not None:
        gp = phase2_result['game_probs']
        print(f"\nGame-level predictions: {len(gp):,}")
        top_picks = gp.nlargest(10, 'hr_prob_game')
        print(f"\nTop 10 HR candidates (test set):")
        print(top_picks[['game_date', 'batter', 'hr_prob_game', 'hit_hr']].to_string(index=False))
    
    return phase1_result, phase2_result


if __name__ == "__main__":
    main()
