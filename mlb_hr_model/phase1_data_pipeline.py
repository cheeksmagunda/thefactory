"""
MLB Home Run Prop Model — Phase 1: Data Pipeline & Feature Engineering
======================================================================

This module handles:
1. Pulling Statcast pitch/PA-level data via pybaseball
2. Engineering ~30 features across batter, pitcher, and environment dimensions
3. Building a clean training dataset where each row = 1 plate appearance

Design philosophy (from GiuseppePaps research):
- Model batter and pitcher INDEPENDENTLY — don't chase batter-vs-arsenal matchups
- Focus on power profile features that are predictive, not descriptive
- Let environment (park + weather) do the contextual heavy lifting
- Avoid overfitting on degrees of freedom

Data sources:
- Statcast (via pybaseball): pitch-level and PA-level batted ball data
- FanGraphs (via pybaseball): season splits, advanced rates
- BallparkPal: hitter-level park factors (manual CSV import from subscription)
- Weather: embedded in BallparkPal factors OR via external API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Date ranges for training/validation/test
    "train_start": "2023-03-30",
    "train_end": "2024-09-30",
    "val_start": "2025-03-20",
    "val_end": "2025-06-30",
    
    # Feature engineering windows
    "recent_form_games": 15,       # Rolling window for recent form
    "season_baseline_min_pa": 50,  # Min PAs before using season stats
    
    # HR model thresholds
    "min_pa_for_batter": 100,      # Min career PAs to include batter
    "min_ip_for_pitcher": 30,      # Min career IP to include pitcher
    
    # Paths
    "data_dir": "data/",
    "output_dir": "output/",
    "ballparkpal_dir": "data/ballparkpal/",  # For manual BPP CSV imports
}


# ============================================================================
# 1. DATA ACQUISITION — Statcast via pybaseball
# ============================================================================

def pull_statcast_data(start_date: str, end_date: str, cache_path: str = None) -> pd.DataFrame:
    """
    Pull Statcast pitch-level data for a date range.
    Caches to parquet for fast reload.
    
    Each row = 1 pitch, but we'll filter to PA-ending events for the HR model.
    """
    from pybaseball import statcast
    
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)
    
    print(f"Pulling Statcast data from {start_date} to {end_date}...")
    print("(This can take 10-30 min for a full season — be patient)")
    
    # pybaseball pulls in ~2-week chunks internally
    df = statcast(start_dt=start_date, end_dt=end_date)
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path)
        print(f"Cached {len(df):,} pitches to {cache_path}")
    
    return df


def filter_to_plate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter Statcast pitch data to PA-ending events only.
    Each row becomes 1 plate appearance with the final outcome.
    """
    # PA-ending event types in Statcast
    pa_events = df[df['events'].notna()].copy()
    
    # Create binary HR target
    pa_events['is_hr'] = (pa_events['events'] == 'home_run').astype(int)
    
    # Create broader outcome categories (useful for future expansion)
    hit_types = ['single', 'double', 'triple', 'home_run']
    pa_events['is_hit'] = pa_events['events'].isin(hit_types).astype(int)
    pa_events['is_xbh'] = pa_events['events'].isin(['double', 'triple', 'home_run']).astype(int)
    
    print(f"Filtered to {len(pa_events):,} plate appearances")
    print(f"HR rate: {pa_events['is_hr'].mean():.4f} ({pa_events['is_hr'].sum():,} HRs)")
    
    return pa_events


# ============================================================================
# 2. BATTER FEATURE ENGINEERING
# ============================================================================

def engineer_batter_features(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer batter-side features for HR prediction.
    
    Core power profile features (from our research synthesis):
    - Barrel rate: strongest single predictor of HR ability
    - Exit velocity: raw power indicator
    - Launch angle: optimal HR range is 25-35 degrees
    - HR/FB rate: how often fly balls leave the yard
    - Pull rate on fly balls: pulled FBs have highest HR probability
    - Fly ball rate: more FBs = more HR opportunities
    - ISO (isolated power): SLG - AVG, pure extra-base power
    
    We compute these as ROLLING SEASON AVERAGES up to (but not including) 
    each game date, preventing data leakage.
    """
    df = pa_df.copy()
    
    # Sort chronologically
    df = df.sort_values(['game_date', 'at_bat_number']).reset_index(drop=True)
    
    # --- Batted ball features (from Statcast) ---
    # Many PAs have no batted ball data (walks, strikeouts, HBP) → NaN in 
    # launch_speed/launch_angle. We fill with 0 (not barreled, not a FB, etc.)
    ev = df['launch_speed'].fillna(0)
    la = df['launch_angle'].fillna(0)
    
    df['is_barrel'] = ((ev >= 98) & (la.between(26, 30)) |
                       ((ev >= 99) & (la.between(25, 31))) |
                       ((ev >= 100) & (la.between(24, 33))) |
                       ((ev >= 101) & (la.between(23, 35))) |
                       ((ev >= 102) & (la.between(22, 37))) |
                       ((ev >= 103) & (la.between(21, 39))) |
                       ((ev >= 104) & (la.between(20, 41))) |
                       ((ev >= 105) & (la.between(19, 43))) |
                       ((ev >= 106) & (la.between(18, 45))) |
                       ((ev >= 107) & (la.between(17, 47))) |
                       ((ev >= 108) & (la.between(16, 50)))).astype(int)
    
    df['is_fly_ball'] = (la.between(25, 60)).astype(int)
    df['is_hard_hit'] = (ev >= 95).astype(int)
    
    # Pull side classification (for HR-specific pull power)
    # RHB pulling = hit to left field, LHB pulling = hit to right field
    # hc_x can be NaN when ball not in play — use NaN-safe approach
    hc = df['hc_x']
    df['is_pulled'] = np.where(
        hc.isna(), np.nan,
        np.where(df['stand'] == 'R', (hc < 126).astype(float), (hc > 126).astype(float))
    )
    
    return df


def compute_batter_rolling_stats(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling season-to-date batter stats for each PA.
    Uses expanding window within each season to prevent leakage.
    
    Key insight: we want PRIOR stats only — what was true about this
    batter BEFORE this plate appearance happened.
    """
    df = pa_df.copy()
    df['season'] = pd.to_datetime(df['game_date']).dt.year
    df = df.sort_values(['batter', 'game_date', 'at_bat_number'])
    
    # Group by batter + season for expanding stats
    batter_season = df.groupby(['batter', 'season'])
    
    # --- Season-to-date stats (expanding, shifted by 1 to prevent leakage) ---
    
    # Barrel rate
    df['batter_barrel_rate_szn'] = batter_season['is_barrel'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # Average exit velocity (on contact only)
    df['batter_avg_ev_szn'] = batter_season['launch_speed'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # Average launch angle
    df['batter_avg_la_szn'] = batter_season['launch_angle'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # HR rate per PA
    df['batter_hr_rate_szn'] = batter_season['is_hr'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # FB rate
    df['batter_fb_rate_szn'] = batter_season['is_fly_ball'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # Hard hit rate
    df['batter_hard_hit_rate_szn'] = batter_season['is_hard_hit'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # Pull rate (on contacted balls)
    df['batter_pull_rate_szn'] = batter_season['is_pulled'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # ISO (using hit outcomes)
    df['_is_single'] = (df['events'] == 'single').astype(int)
    df['_is_double'] = (df['events'] == 'double').astype(int)
    df['_is_triple'] = (df['events'] == 'triple').astype(int)
    df['_total_bases'] = df['_is_single'] + 2*df['_is_double'] + 3*df['_is_triple'] + 4*df['is_hr']
    
    # At-bats (exclude walks, HBP, sac bunts/flies for proper AVG/SLG calc)
    ab_events = ['single', 'double', 'triple', 'home_run', 'field_out', 
                 'strikeout', 'grounded_into_double_play', 'force_out',
                 'fielders_choice', 'double_play', 'triple_play',
                 'field_error', 'strikeout_double_play']
    df['is_ab'] = df['events'].isin(ab_events).astype(int)
    
    df['batter_slg_szn'] = batter_season.apply(
        lambda g: (g['_total_bases'].cumsum().shift(1)) / (g['is_ab'].cumsum().shift(1).replace(0, np.nan))
    ).reset_index(level=[0,1], drop=True)
    
    df['batter_avg_szn'] = batter_season.apply(
        lambda g: (g['is_hit'].cumsum().shift(1)) / (g['is_ab'].cumsum().shift(1).replace(0, np.nan))
    ).reset_index(level=[0,1], drop=True)
    
    df['batter_iso_szn'] = df['batter_slg_szn'] - df['batter_avg_szn']
    
    # PA count (for weighting / reliability)
    df['batter_pa_count_szn'] = batter_season.cumcount()
    
    # --- Recent form (last N games) ---
    # This captures hot/cold streaks that books are slow to price
    df['game_id'] = df['game_date'].astype(str) + '_' + df['batter'].astype(str)
    
    # Recent HR rate (last ~15 games worth of PAs, approx 60 PAs)
    recent_window = 60  # ~15 games × 4 PA/game
    df['batter_hr_rate_recent'] = batter_season['is_hr'].apply(
        lambda x: x.rolling(window=recent_window, min_periods=20).mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    df['batter_barrel_rate_recent'] = batter_season['is_barrel'].apply(
        lambda x: x.rolling(window=recent_window, min_periods=20).mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    df['batter_ev_recent'] = batter_season['launch_speed'].apply(
        lambda x: x.rolling(window=recent_window, min_periods=20).mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # Recent form DELTA (recent - season) — positive = heating up
    df['batter_hr_form_delta'] = df['batter_hr_rate_recent'] - df['batter_hr_rate_szn']
    df['batter_barrel_form_delta'] = df['batter_barrel_rate_recent'] - df['batter_barrel_rate_szn']
    
    # Clean up temp columns
    df.drop(columns=['_is_single', '_is_double', '_is_triple', '_total_bases', 
                     'is_ab', 'game_id'], inplace=True, errors='ignore')
    
    print(f"Computed {sum(1 for c in df.columns if c.startswith('batter_'))} batter features")
    return df


# ============================================================================
# 3. PITCHER FEATURE ENGINEERING
# ============================================================================

def compute_pitcher_rolling_stats(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling season-to-date pitcher stats for each PA.
    
    Pitcher features focused on HR vulnerability:
    - HR/PA rate allowed
    - Fly ball rate allowed (more FBs = more HR chances)
    - Hard contact rate allowed
    - Barrel rate allowed
    - Average EV allowed
    - Fastball velocity (proxy for stuff quality)
    
    Per Giuseppe's research: pitcher profile alone (without matching 
    against batter arsenal preferences) is more predictive.
    """
    df = pa_df.copy()
    df['season'] = pd.to_datetime(df['game_date']).dt.year
    df = df.sort_values(['pitcher', 'game_date', 'at_bat_number'])
    
    pitcher_season = df.groupby(['pitcher', 'season'])
    
    # HR rate allowed
    df['pitcher_hr_rate_szn'] = pitcher_season['is_hr'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # FB rate allowed  
    df['pitcher_fb_rate_szn'] = pitcher_season['is_fly_ball'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # Hard hit rate allowed
    df['pitcher_hard_hit_rate_szn'] = pitcher_season['is_hard_hit'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # Barrel rate allowed
    df['pitcher_barrel_rate_szn'] = pitcher_season['is_barrel'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # Avg exit velocity allowed
    df['pitcher_avg_ev_szn'] = pitcher_season['launch_speed'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # PA count for reliability weighting
    df['pitcher_pa_count_szn'] = pitcher_season.cumcount()
    
    # Recent form (rolling)
    recent_window = 80  # ~20 games for a pitcher
    df['pitcher_hr_rate_recent'] = pitcher_season['is_hr'].apply(
        lambda x: x.rolling(window=recent_window, min_periods=30).mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    df['pitcher_barrel_rate_recent'] = pitcher_season['is_barrel'].apply(
        lambda x: x.rolling(window=recent_window, min_periods=30).mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # Pitcher form delta
    df['pitcher_hr_form_delta'] = df['pitcher_hr_rate_recent'] - df['pitcher_hr_rate_szn']
    
    # --- Pitch quality features (from Statcast pitch data) ---
    # Fastball velocity (release_speed on fastball-classified pitches)
    df['is_fastball'] = df['pitch_type'].isin(['FF', 'SI', 'FC']).astype(int)
    df['fb_velo'] = np.where(df['is_fastball'] == 1, df['release_speed'], np.nan)
    
    df['pitcher_fb_velo_szn'] = pitcher_season['fb_velo'].apply(
        lambda x: x.expanding().mean().shift(1)
    ).reset_index(level=[0,1], drop=True)
    
    # Vertical approach angle approximation (from release pos + extension)
    # VAA is a strong predictor per the discussion in your PDF
    # Steeper (more negative) VAA on FB = harder to lift = fewer HRs
    # We approximate from release_pos_z and plate_z
    if 'release_pos_z' in df.columns and 'plate_z' in df.columns:
        # Rough VAA proxy: angle from release point to plate
        # More negative = steeper, less hittable for power
        df['vaa_proxy'] = np.where(
            df['is_fastball'] == 1,
            np.degrees(np.arctan2(
                df['plate_z'] - df['release_pos_z'],
                df['release_extension'].fillna(6.0) + 54.5  # ~60.5 ft mound to plate adjusted for extension
            )),
            np.nan
        )
        df['pitcher_vaa_szn'] = pitcher_season['vaa_proxy'].apply(
            lambda x: x.expanding().mean().shift(1)
        ).reset_index(level=[0,1], drop=True)
    
    df.drop(columns=['is_fastball', 'fb_velo', 'vaa_proxy'], inplace=True, errors='ignore')
    
    print(f"Computed {sum(1 for c in df.columns if c.startswith('pitcher_'))} pitcher features")
    return df


# ============================================================================
# 4. ENVIRONMENT FEATURES
# ============================================================================

def add_environment_features(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add environment/context features.
    
    BallparkPal methodology insight: park factors should be computed at
    the INDIVIDUAL BATTER level based on spray direction tendencies.
    Their model uses a Contact + Park + Weather model vs a Contact-Only 
    model, and the difference IS the park effect.
    
    For V1, we use:
    - Venue ID (categorical, for tree model to learn park effects)
    - Game-day weather from Statcast metadata where available
    - Platoon advantage indicator (L/R matchup)
    - Batting order position (proxy for expected PAs)
    
    BallparkPal subscription data can be merged in as hitter-level
    park factors when available (see load_ballparkpal_factors).
    """
    df = pa_df.copy()
    
    # --- Platoon advantage ---
    # LHB vs RHP or RHB vs LHP = platoon advantage (historically ~15% more HRs)
    df['platoon_advantage'] = (
        ((df['stand'] == 'L') & (df['p_throws'] == 'R')) |
        ((df['stand'] == 'R') & (df['p_throws'] == 'L'))
    ).astype(int)
    
    # --- Venue (categorical for tree models) ---
    # Tree-based models can learn park effects from the venue ID directly
    df['venue_encoded'] = df['home_team'].astype('category').cat.codes
    
    # --- Day/Night ---
    if 'game_type' in df.columns:
        df['is_night_game'] = (df['game_type'] == 'N').astype(int)
    
    # --- Temperature (from Statcast metadata if available) ---
    # Warmer temps = ball carries further = more HRs
    # BallparkPal found ~10% HR increase at 80°F vs 60°F
    if 'temperature' in df.columns:
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
    
    # --- Wind (if available in Statcast data) ---
    if 'wind_speed' in df.columns:
        df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')
    
    # --- Inning / game state features ---
    df['is_early_innings'] = (df['inning'] <= 5).astype(int)  # Starter more likely in
    
    # --- Batting order position (proxy for PA opportunity) ---
    if 'bat_order' in df.columns:
        df['bat_order'] = pd.to_numeric(df['bat_order'], errors='coerce')
    
    print("Added environment features")
    return df


def load_ballparkpal_factors(filepath: str) -> pd.DataFrame:
    """
    Load hitter-level park factors from BallparkPal subscription export.
    
    Expected columns: game_date, batter_name, batter_id, 
    hr_park_factor, hits_park_factor, runs_park_factor
    
    These represent the BallparkPal CPW model output — the percentage 
    boost/reduction in HR probability for this specific hitter in 
    this specific park on this specific day's weather.
    
    If you don't have BallparkPal data yet, this returns None and 
    the model uses venue_encoded as a proxy.
    """
    if not os.path.exists(filepath):
        print(f"BallparkPal file not found at {filepath}")
        print("Model will use venue ID as park factor proxy (still works, just less precise)")
        return None
    
    bpp = pd.read_csv(filepath)
    print(f"Loaded {len(bpp):,} BallparkPal hitter-level park factors")
    return bpp


# ============================================================================
# 5. PRIOR SEASON STATS (for early-season cold start)
# ============================================================================

def compute_prior_season_baselines(pa_df: pd.DataFrame) -> pd.DataFrame:
    """
    For early-season PAs where season stats are unreliable (< 50 PAs),
    blend with prior-season baselines.
    
    This is critical for Opening Day through ~May when current-season
    sample sizes are tiny.
    """
    df = pa_df.copy()
    df['season'] = pd.to_datetime(df['game_date']).dt.year
    
    # Compute full-season stats for each batter-year
    season_stats = df.groupby(['batter', 'season']).agg(
        barrel_rate=('is_barrel', 'mean'),
        avg_ev=('launch_speed', 'mean'),
        avg_la=('launch_angle', 'mean'),
        hr_rate=('is_hr', 'mean'),
        fb_rate=('is_fly_ball', 'mean'),
        hard_hit_rate=('is_hard_hit', 'mean'),
        total_pa=('is_hr', 'count')
    ).reset_index()
    
    # Create prior season lookup (current_season - 1)
    season_stats['next_season'] = season_stats['season'] + 1
    prior = season_stats.rename(columns={
        'barrel_rate': 'batter_barrel_rate_prior',
        'avg_ev': 'batter_avg_ev_prior', 
        'avg_la': 'batter_avg_la_prior',
        'hr_rate': 'batter_hr_rate_prior',
        'fb_rate': 'batter_fb_rate_prior',
        'hard_hit_rate': 'batter_hard_hit_rate_prior',
        'total_pa': 'batter_pa_prior_season'
    })[['batter', 'next_season', 'batter_barrel_rate_prior', 'batter_avg_ev_prior',
        'batter_avg_la_prior', 'batter_hr_rate_prior', 'batter_fb_rate_prior',
        'batter_hard_hit_rate_prior', 'batter_pa_prior_season']]
    
    df = df.merge(prior, left_on=['batter', 'season'], 
                  right_on=['batter', 'next_season'], how='left')
    df.drop(columns=['next_season'], inplace=True, errors='ignore')
    
    # --- Blend current season with prior season ---
    # Use reliability weighting: as PA count grows, lean more on current season
    min_pa = CONFIG['season_baseline_min_pa']
    
    for feat in ['barrel_rate', 'avg_ev', 'avg_la', 'hr_rate', 'fb_rate', 'hard_hit_rate']:
        szn_col = f'batter_{feat}_szn'
        prior_col = f'batter_{feat}_prior'
        blended_col = f'batter_{feat}_blended'
        
        if szn_col in df.columns and prior_col in df.columns:
            weight = (df['batter_pa_count_szn'] / min_pa).clip(0, 1)
            df[blended_col] = (weight * df[szn_col] + 
                              (1 - weight) * df[prior_col])
            # Fall back to season-only when prior is unavailable
            df[blended_col] = df[blended_col].fillna(df[szn_col])
        elif szn_col in df.columns:
            # No prior season data at all — just use season stats
            df[blended_col] = df[szn_col]
    
    print("Added prior-season baselines with reliability blending")
    return df


# ============================================================================
# 6. ASSEMBLE FINAL FEATURE MATRIX
# ============================================================================

# These are the features the model will train on
FEATURE_COLUMNS = [
    # Batter power profile (blended current + prior season)
    'batter_barrel_rate_blended',
    'batter_avg_ev_blended',
    'batter_avg_la_blended', 
    'batter_hr_rate_blended',
    'batter_fb_rate_blended',
    'batter_hard_hit_rate_blended',
    'batter_iso_szn',
    'batter_pull_rate_szn',
    
    # Batter recent form (streak detection)
    'batter_hr_form_delta',
    'batter_barrel_form_delta',
    
    # Batter context
    'batter_pa_count_szn',
    
    # Pitcher HR vulnerability
    'pitcher_hr_rate_szn',
    'pitcher_fb_rate_szn',
    'pitcher_hard_hit_rate_szn',
    'pitcher_barrel_rate_szn',
    'pitcher_avg_ev_szn',
    'pitcher_fb_velo_szn',
    'pitcher_hr_form_delta',
    'pitcher_pa_count_szn',
    
    # Pitcher quality (VAA)
    'pitcher_vaa_szn',
    
    # Environment
    'platoon_advantage',
    'venue_encoded',
    
    # Weather (when available)
    # 'temperature',
    # 'wind_speed',
    
    # BallparkPal hitter-level park factor (when available)
    # 'hr_park_factor',
]

TARGET_COLUMN = 'is_hr'


def assemble_training_data(pa_df: pd.DataFrame) -> tuple:
    """
    Assemble the final feature matrix and target vector.
    Handles missing values and filters to usable rows.
    """
    df = pa_df.copy()
    
    # Add weather columns if they exist
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
    
    # Also check for optional features
    optional = ['temperature', 'wind_speed', 'hr_park_factor']
    for opt in optional:
        if opt in df.columns:
            available_features.append(opt)
    
    print(f"\nUsing {len(available_features)} features:")
    for f in available_features:
        non_null = df[f].notna().sum()
        print(f"  {f}: {non_null:,} non-null ({non_null/len(df)*100:.1f}%)")
    
    # Filter to rows where we have minimum viable features
    # Try blended first, fall back to season-only
    core_features = [
        'batter_barrel_rate_blended', 'batter_hr_rate_blended',
        'pitcher_hr_rate_szn', 'platoon_advantage'
    ]
    core_available = [f for f in core_features if f in df.columns and df[f].notna().any()]
    
    if not core_available:
        # Absolute fallback: use any available batter + pitcher feature
        core_available = [f for f in available_features if df[f].notna().any()][:4]
        print(f"Warning: using fallback core features: {core_available}")
    
    mask = df[core_available].notna().all(axis=1)
    df_clean = df[mask].copy()
    
    print(f"\nRows after filtering: {len(df_clean):,} ({len(df_clean)/len(df)*100:.1f}%)")
    print(f"HR rate in clean data: {df_clean[TARGET_COLUMN].mean():.4f}")
    
    X = df_clean[available_features]
    y = df_clean[TARGET_COLUMN]
    
    # Store metadata for later joining
    meta_cols = ['game_date', 'batter', 'pitcher', 'home_team', 
                 'away_team', 'stand', 'p_throws', 'events']
    meta = df_clean[[c for c in meta_cols if c in df_clean.columns]]
    
    return X, y, meta, available_features


# ============================================================================
# 7. MASTER PIPELINE
# ============================================================================

def run_phase1_pipeline(start_date: str, end_date: str, 
                        cache_path: str = None) -> dict:
    """
    Run the complete Phase 1 pipeline:
    1. Pull Statcast data
    2. Filter to plate appearances
    3. Engineer batter features
    4. Engineer pitcher features  
    5. Add environment features
    6. Add prior-season baselines
    7. Assemble training matrix
    
    Returns dict with X, y, metadata, feature names, and full dataframe.
    """
    print("=" * 60)
    print("MLB HR Model — Phase 1 Pipeline")
    print("=" * 60)
    
    # Step 1: Get data
    raw = pull_statcast_data(start_date, end_date, cache_path)
    
    # Step 2: Filter to PAs
    pa_df = filter_to_plate_appearances(raw)
    
    # Step 3: Batter features
    pa_df = engineer_batter_features(pa_df)
    pa_df = compute_batter_rolling_stats(pa_df)
    
    # Step 4: Pitcher features
    pa_df = compute_pitcher_rolling_stats(pa_df)
    
    # Step 5: Environment
    pa_df = add_environment_features(pa_df)
    
    # Step 6: Prior season baselines
    pa_df = compute_prior_season_baselines(pa_df)
    
    # Step 7: Assemble
    X, y, meta, feature_names = assemble_training_data(pa_df)
    
    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print(f"Training matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"HR rate: {y.mean():.4f} ({y.sum():,} HRs)")
    print(f"Date range: {meta['game_date'].min()} to {meta['game_date'].max()}")
    print("=" * 60)
    
    return {
        'X': X,
        'y': y,
        'meta': meta,
        'feature_names': feature_names,
        'full_df': pa_df
    }


# ============================================================================
# QUICK START (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example: pull a small sample to test the pipeline
    # For full training, use the CONFIG dates above
    
    print("Testing pipeline with a small date range...")
    print("For full runs, use run_phase1_pipeline() with CONFIG dates")
    print()
    
    # You would run:
    # result = run_phase1_pipeline(
    #     CONFIG['train_start'], 
    #     CONFIG['train_end'],
    #     cache_path='data/statcast_2023_2024.parquet'
    # )
    
    print("Pipeline module loaded successfully!")
    print(f"Feature set: {len(FEATURE_COLUMNS)} features defined")
    print(f"\nTo run: import phase1_data_pipeline as p1")
    print(f"        result = p1.run_phase1_pipeline('2023-03-30', '2024-09-30')")
