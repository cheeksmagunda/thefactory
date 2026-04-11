"""
MLB Home Run Prop Model — Phase 2: Model Training & Calibration
================================================================

This module handles:
1. Train/validation/test splitting (temporal, not random)
2. LightGBM training with HR-optimized hyperparameters
3. Isotonic regression calibration (per Monte Carlo model findings)
4. Model evaluation (calibration curves, ROI simulation, feature importance)
5. PA-level → game-level HR probability aggregation

Key design decisions:
- Temporal splits only (no random splitting — would leak future data)
- Isotonic calibration > Platt scaling (confirmed by Monte Carlo backtest)
- Confidence tiering for bet sizing (top 20% by edge → 3x better ROI)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score, 
    precision_recall_curve, average_precision_score
)
import json
import os
import pickle


# ============================================================================
# 1. TEMPORAL TRAIN/VAL/TEST SPLIT
# ============================================================================

def temporal_split(X, y, meta, 
                   train_end='2024-06-30',
                   val_end='2024-09-30'):
    """
    Split data temporally — NEVER randomly for time-series sports data.
    
    Train:  everything before train_end
    Val:    train_end to val_end (for calibration + hyperparam tuning)
    Test:   everything after val_end (holdout for final eval)
    """
    dates = pd.to_datetime(meta['game_date'])
    
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    splits = {
        'train': (X[train_mask], y[train_mask], meta[train_mask]),
        'val': (X[val_mask], y[val_mask], meta[val_mask]),
        'test': (X[test_mask], y[test_mask], meta[test_mask]),
    }
    
    for name, (Xi, yi, mi) in splits.items():
        hr_rate = yi.mean() if len(yi) > 0 else 0
        print(f"{name:6s}: {len(Xi):>8,} PAs | {yi.sum():>5,} HRs | "
              f"HR rate: {hr_rate:.4f} | "
              f"{mi['game_date'].min()} to {mi['game_date'].max()}")
    
    return splits


# ============================================================================
# 2. LIGHTGBM TRAINING
# ============================================================================

def get_lgb_params():
    """
    LightGBM hyperparameters tuned for HR prediction.
    
    Key considerations:
    - HR is a rare event (~3% of PAs) → use scale_pos_weight or focal loss
    - We want well-calibrated probabilities, not just ranking → logloss objective
    - Moderate depth to avoid overfitting on sparse matchup data
    - Low learning rate + more trees for stability
    """
    return {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        
        # Tree structure
        'num_leaves': 48,            # Moderate complexity
        'max_depth': 6,              # Prevent overfitting
        'min_child_samples': 50,     # Balanced for rare events
        
        # Regularization
        'lambda_l1': 0.05,           # L1 regularization
        'lambda_l2': 0.5,            # L2 regularization
        'min_gain_to_split': 0.001,  # Lower threshold — let trees grow
        'feature_fraction': 0.8,     # Column subsampling
        'bagging_fraction': 0.8,     # Row subsampling
        'bagging_freq': 5,
        
        # Learning
        'learning_rate': 0.05,       # Higher LR — was too low before
        'n_estimators': 2000,        # Will early-stop well before this
        
        # Class imbalance (HRs are ~3% of PAs)
        # OLD: scale_pos_weight=30 caused first tree to overfit → immediate early stop
        # NEW: light weighting lets the model build hundreds of trees
        'scale_pos_weight': 3,       # Gentle boost for positive class
        'is_unbalance': False,
        
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
    }


def train_model(X_train, y_train, X_val, y_val, feature_names=None):
    """
    Train LightGBM with early stopping on validation set.
    """
    params = get_lgb_params()
    n_estimators = params.pop('n_estimators')
    
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        **params
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )
    
    print(f"\nBest iteration: {model.best_iteration_}")
    print(f"Best val logloss: {model.best_score_['valid_0']['binary_logloss']:.6f}")
    
    return model


# ============================================================================
# 3. ISOTONIC CALIBRATION
# ============================================================================

def calibrate_model(model, X_cal, y_cal):
    """
    Apply isotonic regression calibration to model probabilities.
    
    Why isotonic > Platt scaling for HR props:
    - Platt scaling assumes a sigmoid relationship between raw scores 
      and true probabilities — too rigid for the extreme tails
    - HR probabilities live in the 1-8% range where small calibration 
      errors compound into large edge miscalculations
    - Isotonic regression is non-parametric — fits any monotonic shape
    - The Monte Carlo model author confirmed isotonic helped 
      "enormously in the tails"
    
    Uses the validation set for fitting (NOT training set).
    """
    raw_probs = model.predict_proba(X_cal)[:, 1]
    
    calibrator = IsotonicRegression(
        y_min=0.001,   # Floor — no PA has 0% HR chance
        y_max=0.25,    # Cap — even the best matchups rarely exceed 15-20%
        out_of_bounds='clip'
    )
    calibrator.fit(raw_probs, y_cal)
    
    calibrated_probs = calibrator.predict(raw_probs)
    
    # Report calibration improvement
    raw_brier = brier_score_loss(y_cal, raw_probs)
    cal_brier = brier_score_loss(y_cal, calibrated_probs)
    
    print(f"\nCalibration results on validation set:")
    print(f"  Raw Brier score:        {raw_brier:.6f}")
    print(f"  Calibrated Brier score: {cal_brier:.6f}")
    print(f"  Improvement:            {(raw_brier - cal_brier)/raw_brier*100:.1f}%")
    
    return calibrator


def get_calibrated_probs(model, calibrator, X):
    """Get calibrated HR probabilities for a feature matrix."""
    raw_probs = model.predict_proba(X)[:, 1]
    return calibrator.predict(raw_probs)


# ============================================================================
# 4. PA-LEVEL → GAME-LEVEL AGGREGATION
# ============================================================================

def aggregate_to_game_level(probs, meta, expected_pa=4.0):
    """
    Convert PA-level HR probabilities to game-level predictions.
    
    P(at least 1 HR in game) = 1 - P(no HR in any PA)
    = 1 - ∏(1 - p_i) for each PA
    
    For pre-game predictions, we estimate expected PAs based on
    lineup position and projected game total.
    
    For backtesting, we use actual PAs that occurred.
    """
    game_df = pd.DataFrame({
        'game_date': meta['game_date'].values,
        'batter': meta['batter'].values,
        'pitcher': meta.get('pitcher', pd.Series([None]*len(meta))).values,
        'hr_prob_pa': probs,
        'actual_hr': meta.get('is_hr', pd.Series([0]*len(meta))).values if 'is_hr' in meta.columns else 0,
    })
    
    # Aggregate by batter + game
    game_agg = game_df.groupby(['game_date', 'batter']).agg(
        # Probability of at least 1 HR = 1 - product of (1 - p) across all PAs
        hr_prob_game=('hr_prob_pa', lambda x: 1 - np.prod(1 - x)),
        avg_pa_prob=('hr_prob_pa', 'mean'),
        num_pa=('hr_prob_pa', 'count'),
        actual_hrs=('actual_hr', 'sum'),
    ).reset_index()
    
    game_agg['hit_hr'] = (game_agg['actual_hrs'] >= 1).astype(int)
    
    return game_agg


# ============================================================================
# 5. EDGE DETECTION & CONFIDENCE TIERS
# ============================================================================

def compute_edges(game_probs_df, odds_df=None):
    """
    Compare model probabilities against market implied probabilities.
    
    Edge = model_prob - market_implied_prob
    
    If no odds data provided, we can still rank by raw probability
    (useful for identifying the best HR candidates each day).
    
    Market implied probability from American odds:
    - Positive odds (+400): implied = 100 / (odds + 100) = 0.20
    - Negative odds (-150): implied = abs(odds) / (abs(odds) + 100) = 0.60
    """
    df = game_probs_df.copy()
    
    if odds_df is not None:
        # Merge market odds
        df = df.merge(odds_df, on=['game_date', 'batter'], how='left')
        
        if 'market_implied_prob' in df.columns:
            df['edge'] = df['hr_prob_game'] - df['market_implied_prob']
            df['edge_pct'] = df['edge'] / df['market_implied_prob'] * 100
        elif 'american_odds' in df.columns:
            df['market_implied_prob'] = np.where(
                df['american_odds'] > 0,
                100 / (df['american_odds'] + 100),
                np.abs(df['american_odds']) / (np.abs(df['american_odds']) + 100)
            )
            # Remove vig estimate (~5% for HR props)
            df['market_implied_prob_novig'] = df['market_implied_prob'] * 0.95
            df['edge'] = df['hr_prob_game'] - df['market_implied_prob_novig']
            df['edge_pct'] = df['edge'] / df['market_implied_prob_novig'] * 100
    
    # Confidence tiers based on edge size (or raw probability if no odds)
    sort_col = 'edge' if 'edge' in df.columns else 'hr_prob_game'
    df['rank_pct'] = df[sort_col].rank(pct=True)
    
    df['confidence_tier'] = pd.cut(
        df['rank_pct'],
        bins=[0, 0.6, 0.8, 0.9, 1.0],
        labels=['D - Skip', 'C - Monitor', 'B - Lean', 'A - Strong']
    )
    
    return df


def american_odds_to_implied(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def implied_to_american(prob):
    """Convert implied probability to American odds."""
    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    else:
        return 100 * (1 - prob) / prob


# ============================================================================
# 6. EVALUATION METRICS
# ============================================================================

def evaluate_model(y_true, y_probs, label="Model"):
    """
    Comprehensive evaluation for HR probability model.
    
    Key metrics:
    - Brier score: overall probability accuracy
    - ECE: expected calibration error (are probabilities trustworthy?)
    - AUC-ROC: discrimination ability
    - Calibration curve: visual check
    """
    results = {}
    
    # Brier score (lower = better)
    results['brier_score'] = brier_score_loss(y_true, y_probs)
    
    # Log loss
    y_probs_clipped = np.clip(y_probs, 1e-7, 1 - 1e-7)
    results['log_loss'] = log_loss(y_true, y_probs_clipped)
    
    # AUC-ROC
    results['auc_roc'] = roc_auc_score(y_true, y_probs)
    
    # Expected Calibration Error (ECE)
    n_bins = 10
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_probs, n_bins=n_bins, strategy='quantile'
    )
    ece = np.mean(np.abs(fraction_of_positives - mean_predicted))
    results['ece'] = ece
    
    # Average precision (better than AUC for imbalanced)
    results['avg_precision'] = average_precision_score(y_true, y_probs)
    
    print(f"\n{'='*50}")
    print(f"Evaluation: {label}")
    print(f"{'='*50}")
    print(f"  Brier Score:     {results['brier_score']:.6f}")
    print(f"  Log Loss:        {results['log_loss']:.6f}")
    print(f"  AUC-ROC:         {results['auc_roc']:.4f}")
    print(f"  ECE:             {results['ece']:.4f} ({results['ece']*100:.1f}%)")
    print(f"  Avg Precision:   {results['avg_precision']:.4f}")
    print(f"  Base HR rate:    {y_true.mean():.4f}")
    print(f"  Mean pred:       {y_probs.mean():.4f}")
    
    return results, (fraction_of_positives, mean_predicted)


def simulate_betting_roi(edges_df, stake=1.0):
    """
    Simulate flat-stake betting ROI across confidence tiers.
    
    For each bet where we have a positive edge:
    - Stake $1 at the given odds
    - Track wins/losses
    - Compute ROI by tier
    
    The Monte Carlo model showed:
    - All tiers: +2.1% ROI
    - Top 20% (Tier A): +6.1% ROI
    """
    if 'edge' not in edges_df.columns or 'american_odds' not in edges_df.columns:
        print("Need edge and odds columns for ROI simulation")
        return None
    
    df = edges_df[edges_df['edge'] > 0].copy()
    
    if len(df) == 0:
        print("No positive edge bets found")
        return None
    
    # Calculate payout per bet
    df['payout'] = np.where(
        df['american_odds'] > 0,
        stake * df['american_odds'] / 100,
        stake * 100 / np.abs(df['american_odds'])
    )
    
    df['profit'] = np.where(
        df['hit_hr'] == 1,
        df['payout'],          # Win: get payout
        -stake                 # Lose: lose stake
    )
    
    # Overall ROI
    total_wagered = len(df) * stake
    total_profit = df['profit'].sum()
    overall_roi = total_profit / total_wagered * 100
    
    print(f"\n{'='*50}")
    print(f"Betting Simulation (flat ${stake} stakes)")
    print(f"{'='*50}")
    print(f"Total bets:    {len(df):,}")
    print(f"Total wagered: ${total_wagered:,.0f}")
    print(f"Total profit:  ${total_profit:,.2f}")
    print(f"Overall ROI:   {overall_roi:+.2f}%")
    print(f"Win rate:      {df['hit_hr'].mean()*100:.1f}%")
    
    # ROI by confidence tier
    print(f"\nBy confidence tier:")
    for tier in ['A - Strong', 'B - Lean', 'C - Monitor', 'D - Skip']:
        tier_df = df[df['confidence_tier'] == tier]
        if len(tier_df) > 0:
            tier_roi = tier_df['profit'].sum() / (len(tier_df) * stake) * 100
            print(f"  {tier}: {len(tier_df):>5} bets | "
                  f"ROI: {tier_roi:+.2f}% | "
                  f"Win: {tier_df['hit_hr'].mean()*100:.1f}%")
    
    return df


def get_feature_importance(model, feature_names):
    """Extract and display feature importance from trained model."""
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance['importance_pct'] = (importance['importance'] / 
                                    importance['importance'].sum() * 100)
    
    print(f"\nTop 15 features by importance:")
    for _, row in importance.head(15).iterrows():
        bar = '█' * int(row['importance_pct'])
        print(f"  {row['feature']:35s} {bar} {row['importance_pct']:.1f}%")
    
    return importance


# ============================================================================
# 7. SAVE / LOAD MODEL
# ============================================================================

def save_model(model, calibrator, feature_names, path='models/'):
    """Save trained model, calibrator, and config."""
    os.makedirs(path, exist_ok=True)
    
    model.booster_.save_model(f'{path}/lgb_hr_model.txt')
    
    with open(f'{path}/calibrator.pkl', 'wb') as f:
        pickle.dump(calibrator, f)
    
    with open(f'{path}/config.json', 'w') as f:
        json.dump({
            'feature_names': feature_names,
            'trained_at': str(pd.Timestamp.now()),
            'lgb_params': get_lgb_params(),
        }, f, indent=2)
    
    print(f"Model saved to {path}/")


def load_model(path='models/'):
    """Load trained model and calibrator."""
    model = lgb.Booster(model_file=f'{path}/lgb_hr_model.txt')
    
    with open(f'{path}/calibrator.pkl', 'rb') as f:
        calibrator = pickle.load(f)
    
    with open(f'{path}/config.json', 'r') as f:
        config = json.load(f)
    
    return model, calibrator, config


# ============================================================================
# 8. MASTER TRAINING PIPELINE  
# ============================================================================

def run_phase2_pipeline(phase1_result: dict,
                        train_end='2024-06-30',
                        val_end='2024-09-30') -> dict:
    """
    Run the complete Phase 2 pipeline:
    1. Temporal split
    2. Train LightGBM
    3. Calibrate with isotonic regression
    4. Evaluate on test set
    5. Aggregate to game level
    6. Feature importance analysis
    
    Returns trained model, calibrator, and evaluation results.
    """
    X = phase1_result['X']
    y = phase1_result['y']
    meta = phase1_result['meta']
    feature_names = phase1_result['feature_names']
    
    # Fill NaN for LightGBM (it handles NaN natively, but empty cols cause issues)
    X = X.copy()
    
    print("=" * 60)
    print("MLB HR Model — Phase 2: Training & Calibration")
    print("=" * 60)
    print(f"Total data: {len(X):,} rows, {len(feature_names)} features")
    print(f"HR rate: {y.mean():.4f} ({y.sum():,} HRs)")
    
    # Step 1: Split — try temporal, fall back to percentage if splits are empty
    splits = temporal_split(X, y, meta, train_end, val_end)
    X_train, y_train, meta_train = splits['train']
    X_val, y_val, meta_val = splits['val']
    X_test, y_test, meta_test = splits['test']
    
    # If val or test are empty, use percentage-based split instead
    if len(X_val) < 50 or len(X_test) < 50:
        print("\nTemporal split produced too-small val/test — using 60/20/20 split")
        n = len(X)
        idx = np.arange(n)
        # Keep temporal ordering — don't shuffle
        t1, t2 = int(n * 0.6), int(n * 0.8)
        X_train, y_train = X.iloc[idx[:t1]], y.iloc[idx[:t1]]
        meta_train = meta.iloc[idx[:t1]]
        X_val, y_val = X.iloc[idx[t1:t2]], y.iloc[idx[t1:t2]]
        meta_val = meta.iloc[idx[t1:t2]]
        X_test, y_test = X.iloc[idx[t2:]], y.iloc[idx[t2:]]
        meta_test = meta.iloc[idx[t2:]]
        
        print(f"  train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}")
    
    # Step 2: Train
    print("\n--- Training LightGBM ---")
    model = train_model(X_train, y_train, X_val, y_val, feature_names)
    
    # Step 3: Calibrate (on validation set)
    print("\n--- Isotonic Calibration ---")
    calibrator = calibrate_model(model, X_val, y_val)
    
    # Step 4: Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_probs_raw = model.predict_proba(X_test)[:, 1]
    test_probs_cal = calibrator.predict(test_probs_raw)
    
    print("\nRaw model:")
    evaluate_model(y_test.values, test_probs_raw, "Raw LightGBM (Test)")
    
    print("\nCalibrated model:")
    eval_results, cal_curve = evaluate_model(
        y_test.values, test_probs_cal, "Calibrated LightGBM (Test)"
    )
    
    # Step 5: Aggregate to game level
    print("\n--- Game-Level Aggregation ---")
    meta_test_with_hr = meta_test.copy()
    meta_test_with_hr['is_hr'] = y_test.values
    game_probs = aggregate_to_game_level(test_probs_cal, meta_test_with_hr)
    
    print(f"Game-level predictions: {len(game_probs):,}")
    print(f"Mean game HR prob: {game_probs['hr_prob_game'].mean():.4f}")
    print(f"Actual HR rate (game): {game_probs['hit_hr'].mean():.4f}")
    
    # Step 6: Feature importance
    print("\n--- Feature Importance ---")
    feat_imp = get_feature_importance(model, feature_names)
    
    # Step 7: Edge detection (without odds for now)
    edges = compute_edges(game_probs)
    
    # Save everything
    save_model(model, calibrator, feature_names)
    
    return {
        'model': model,
        'calibrator': calibrator,
        'eval_results': eval_results,
        'cal_curve': cal_curve,
        'game_probs': game_probs,
        'edges': edges,
        'feature_importance': feat_imp,
        'feature_names': feature_names,
    }


if __name__ == "__main__":
    print("Phase 2 module loaded successfully!")
    print("Run after Phase 1: result2 = run_phase2_pipeline(phase1_result)")
