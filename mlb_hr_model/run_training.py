"""
MLB HR Model — Training Integration
====================================

Wires together all feature sources into a single training pipeline:

  1. phase1_data_pipeline.py  → Base Statcast features (22 features)
  2. advanced_features.py     → Advanced Statcast features (35+ features)
  3. ballparkpal_integration.py → BPP cross-reference features (12 features)

Run flow:
  python run_training.py --start 2023-03-30 --end 2025-09-30

This will:
  1. Pull Statcast data via pybaseball
  2. Build all feature layers (standard + advanced)
  3. Merge BallparkPal features if export files exist in data/ballparkpal/
  4. Train LightGBM with isotonic calibration
  5. Output trained model + evaluation metrics
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase1_data_pipeline import (
    pull_statcast_data, engineer_batter_features,
    add_environment_features, assemble_training_data,
    FEATURE_COLUMNS as BASE_FEATURES
)
from advanced_features import engineer_advanced_features
from ballparkpal_integration import extract_bpp_features, load_bpp_export
from phase2_model_training import (
    temporal_split, get_lgb_params, train_model,
    calibrate_model, evaluate_model, get_calibrated_probs,
    save_model,
)


# ============================================================================
# COMBINED FEATURE SET
# ============================================================================

# BallparkPal features (only available when export is provided)
BPP_FEATURES = [
    'bpp_hr_prob', 'bpp_hr_prob_raw', 'bpp_hr_boost',
    'bpp_vs_grade', 'bpp_rc', 'bpp_k_prob',
    'bpp_park_impact', 'bpp_contact_adj_hr',
    'bpp_power_prob', 'bpp_validation_score',
    'bpp_bvp_hr', 'bpp_bvp_pa',
]

# ALL_FEATURES is built dynamically after advanced features are computed.
# BASE_FEATURES come from phase1, advanced features are discovered at runtime.
ALL_FEATURES = list(set(BASE_FEATURES + BPP_FEATURES))


# ============================================================================
# INTEGRATED PIPELINE
# ============================================================================

def run_integrated_pipeline(start_date='2023-03-30', end_date='2025-09-30',
                             bpp_dir=None, quick_test=False):
    """
    Full integrated training pipeline.
    
    Args:
        start_date: Training data start
        end_date: Training data end
        bpp_dir: Directory with BallparkPal export files (optional)
        quick_test: If True, only pull 30 days of data
    """
    print("=" * 70)
    print("  MLB HR MODEL — INTEGRATED TRAINING PIPELINE")
    print("=" * 70)
    
    # --- Step 1: Pull Statcast data ---
    print("\n[1/5] Pulling Statcast data...")
    if quick_test:
        from datetime import datetime, timedelta
        end = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = (end - timedelta(days=30)).strftime('%Y-%m-%d')
    
    pa_df = pull_statcast_data(start_date, end_date)
    print(f"  Pulled {len(pa_df):,} PAs from {start_date} to {end_date}")
    
    # --- Step 2: Build base features (Phase 1) ---
    print("\n[2/5] Engineering base features...")
    pa_df = engineer_batter_features(pa_df)
    pa_df = add_environment_features(pa_df)
    print(f"  Base features: {len(BASE_FEATURES)} columns")
    
    # --- Step 3: Build advanced features ---
    print("\n[3/5] Engineering advanced features...")
    advanced_df = engineer_advanced_features(pa_df)
    # Merge advanced features back into the main DataFrame
    for col in advanced_df.columns:
        pa_df[col] = advanced_df[col].values
    advanced_feature_names = list(advanced_df.columns)
    # Update ALL_FEATURES with the dynamically discovered advanced feature names
    global ALL_FEATURES
    ALL_FEATURES = list(set(BASE_FEATURES + advanced_feature_names + BPP_FEATURES))
    print(f"  Advanced features: {len(advanced_feature_names)} columns")
    
    # --- Step 4: Merge BallparkPal data (if available) ---
    bpp_features_added = False
    if bpp_dir and os.path.isdir(bpp_dir):
        print(f"\n[3.5/5] Loading BallparkPal exports from {bpp_dir}...")
        bpp_files = sorted([f for f in os.listdir(bpp_dir) 
                           if f.startswith('Matchups_') and f.endswith('.xlsx')])
        
        if bpp_files:
            bpp_dfs = []
            for bf in bpp_files:
                try:
                    bpp_raw = load_bpp_export(os.path.join(bpp_dir, bf))
                    bpp_feat = extract_bpp_features(bpp_raw)
                    # Extract date from filename
                    date_str = bf.replace('Matchups_', '').replace('.xlsx', '')
                    bpp_feat['game_date'] = pd.to_datetime(date_str)
                    bpp_dfs.append(bpp_feat)
                except Exception as e:
                    print(f"  Warning: Failed to load {bf}: {e}")
            
            if bpp_dfs:
                bpp_all = pd.concat(bpp_dfs, ignore_index=True)
                print(f"  Loaded {len(bpp_dfs)} BPP exports with {len(bpp_all)} hitter rows")
                
                # Merge on batter name + game date (approximate match)
                # In production, this would use player IDs
                pa_df['game_date'] = pd.to_datetime(pa_df['game_date'])
                pa_df = pa_df.merge(
                    bpp_all, 
                    left_on=['game_date', 'player_name'],
                    right_on=['game_date', 'Batter'],
                    how='left',
                    suffixes=('', '_bpp')
                )
                bpp_features_added = True
                print(f"  Merged BPP features. Match rate: {pa_df['bpp_hr_prob'].notna().mean()*100:.1f}%")
    
    if not bpp_features_added:
        print("\n  [Note] No BallparkPal data available. Training on Statcast features only.")
        print("  To include BPP: save exports to data/ballparkpal/ and re-run with --bpp data/ballparkpal/")
    
    # --- Step 5: Assemble and train ---
    print("\n[4/5] Assembling training data...")
    
    # Use all available features
    available = [f for f in ALL_FEATURES if f in pa_df.columns and pa_df[f].notna().sum() > 100]
    print(f"  Available features: {len(available)} of {len(ALL_FEATURES)} total")
    
    # Filter to PA-ending events
    pa_events = pa_df[pa_df['events'].notna()].copy()
    pa_events['is_hr'] = (pa_events['events'] == 'home_run').astype(int)
    
    X = pa_events[available].fillna(0)
    y = pa_events['is_hr']
    
    print(f"  Training set: {len(X):,} PAs, {y.sum():,} HRs ({y.mean()*100:.2f}%)")
    print(f"  Features: {X.shape[1]}")
    
    # Temporal split
    print("\n[5/5] Training model...")
    pa_events['game_date'] = pd.to_datetime(pa_events['game_date'])

    # Build metadata DataFrame for temporal_split
    meta_cols = ['game_date', 'batter', 'pitcher', 'home_team',
                 'away_team', 'stand', 'p_throws', 'events']
    meta = pa_events[[c for c in meta_cols if c in pa_events.columns]]

    splits = temporal_split(X, y, meta)

    X_train, y_train, meta_train = splits['train']
    X_val, y_val, meta_val = splits['val']
    X_test, y_test, meta_test = splits['test']

    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Train
    model = train_model(X_train, y_train, X_val, y_val, feature_names=available)

    # Calibrate
    cal_model = calibrate_model(model, X_val, y_val)

    # Evaluate
    test_probs_cal = get_calibrated_probs(model, cal_model, X_test)
    metrics, _ = evaluate_model(y_test.values, test_probs_cal, "Integrated Test")

    # Save using phase2's save_model
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    save_model(model, cal_model, available, path=model_dir)
    
    print(f"\n  Model saved to {model_dir}/")
    print(f"  Features used: {len(available)}")
    
    return model, cal_model, metrics


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MLB HR Model — Integrated Training')
    parser.add_argument('--start', default='2023-03-30', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-09-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--bpp', default=None, help='BallparkPal exports directory')
    parser.add_argument('--test', action='store_true', help='Quick test (30 days)')
    args = parser.parse_args()
    
    run_integrated_pipeline(
        start_date=args.start,
        end_date=args.end,
        bpp_dir=args.bpp,
        quick_test=args.test,
    )
