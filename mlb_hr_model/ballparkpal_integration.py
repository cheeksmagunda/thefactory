"""
MLB HR Model — BallparkPal Integration Layer
=============================================

Cross-references our Statcast features with BallparkPal simulation data.

PHILOSOPHY:
  Advanced Statcast features = discovery engine (finds matchups)
  BallparkPal simulation    = reality check (confirms the hitter 
                               actually has a chance TODAY)

Cross-reference tiers:
  ELITE:        HR Prob >= 3.5 AND HR Prob (no park) >= 3.5 AND vs Grade >= 5 AND K Prob <= 22%
  VALIDATED:    HR Prob >= 3.0 AND HR Prob (no park) >= 3.0 AND vs Grade >= 5
  MATCHUP_ONLY: HR Prob (no park) >= 3.5 AND vs Grade >= 5 but environment suppressing
  SUPPRESSED:   HR Prob (no park) >= 4.0 AND HR Boost <= -15% (series persistence watch)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def load_bpp_export(filepath):
    """Load and validate a BallparkPal matchups export."""
    df = pd.read_excel(filepath)
    required = ['Batter', 'Pitcher', 'HR Prob', 'HR Prob (no park)', 'HR Boost', 'vs Grade', 'Starter']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for col in ['HR Prob', 'HR Prob (no park)', 'HR Boost', 'vs Grade', 'RC', 'K Prob']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['Starter'] = pd.to_numeric(df['Starter'], errors='coerce').fillna(0).astype(int)

    # Derived
    df['park_delta'] = df['HR Prob'] - df['HR Prob (no park)']
    df['k_adjusted_hr'] = df['HR Prob'] * (1 - df['K Prob'] / 100)
    df['bpp_validation_score'] = (
        (df['HR Prob'] >= 3.5).astype(int) +
        (df['HR Prob (no park)'] >= 3.5).astype(int) +
        (df['vs Grade'] >= 5).astype(int) +
        (df['HR Boost'] >= 0).astype(int)
    )
    print(f"Loaded BPP: {len(df)} hitters, {df['Game'].nunique()} games, {df['Starter'].sum()} starters")
    return df


def extract_bpp_features(bpp_df):
    """Extract model-ready features from BPP data."""
    f = bpp_df[['Game', 'Team', 'Batter', 'Pitcher', 'Starter']].copy()
    f['bpp_hr_prob'] = bpp_df['HR Prob'] / 100
    f['bpp_hr_prob_raw'] = bpp_df['HR Prob (no park)'] / 100
    f['bpp_hr_boost'] = bpp_df['HR Boost'] / 100
    f['bpp_vs_grade'] = bpp_df['vs Grade']
    f['bpp_rc'] = bpp_df['RC']
    f['bpp_k_prob'] = bpp_df['K Prob'] / 100
    f['bpp_park_impact'] = bpp_df['park_delta'] / 100
    f['bpp_contact_adj_hr'] = (bpp_df['HR Prob'] / 100) * (1 - bpp_df['K Prob'] / 100) * 4
    f['bpp_validation_score'] = bpp_df['bpp_validation_score']
    if 'XB Prob' in bpp_df.columns:
        f['bpp_power_prob'] = (bpp_df['HR Prob'] + bpp_df['XB Prob']) / 100
    f['bpp_bvp_hr'] = pd.to_numeric(bpp_df.get('HR (hist)', 0), errors='coerce').fillna(0).astype(int)
    f['bpp_bvp_pa'] = pd.to_numeric(bpp_df.get('PA', 0), errors='coerce').fillna(0).astype(int)
    return f


def generate_pick_universe(bpp_filepath, model_scores=None):
    """Cross-reference BPP with model scores to produce tiered pick universe."""
    bpp = load_bpp_export(bpp_filepath)
    starters = bpp[bpp['Starter'] == 1].copy()
    starters['tier'] = 'SKIP'

    # ELITE: high HR + low K + confirmed environment + good grade
    elite_mask = (
        (starters['HR Prob'] >= 3.5) &
        (starters['HR Prob (no park)'] >= 3.5) &
        (starters['vs Grade'] >= 5) &
        (starters['K Prob'] <= 22)
    )
    starters.loc[elite_mask, 'tier'] = 'ELITE'

    # VALIDATED: good HR prob + confirmed environment + good grade (higher K ok)
    validated_mask = (
        (starters['HR Prob'] >= 3.0) &
        (starters['HR Prob (no park)'] >= 3.0) &
        (starters['vs Grade'] >= 5) &
        (starters['HR Boost'] >= -10) &
        (starters['tier'] == 'SKIP')
    )
    starters.loc[validated_mask, 'tier'] = 'VALIDATED'

    # MATCHUP_ONLY: great raw matchup, environment neutral-to-negative
    matchup_mask = (
        (starters['HR Prob (no park)'] >= 3.5) &
        (starters['vs Grade'] >= 5) &
        (starters['tier'] == 'SKIP')
    )
    starters.loc[matchup_mask, 'tier'] = 'MATCHUP_ONLY'

    # SUPPRESSED: elite matchup being killed by park
    suppressed_mask = (
        (starters['HR Prob (no park)'] >= 4.0) &
        (starters['HR Boost'] <= -15) &
        (starters['tier'] == 'SKIP')
    )
    starters.loc[suppressed_mask, 'tier'] = 'SUPPRESSED'

    if model_scores:
        starters['model_danger_score'] = starters['Batter'].map(model_scores).fillna(0)

    tier_order = {'ELITE': 0, 'VALIDATED': 1, 'MATCHUP_ONLY': 2, 'SUPPRESSED': 3, 'SKIP': 4}
    starters['tier_rank'] = starters['tier'].map(tier_order)
    starters = starters.sort_values(['tier_rank', 'HR Prob'], ascending=[True, False])
    return starters


def print_pick_universe(picks_df, top_n=25):
    """Pretty-print the tiered pick universe."""
    print(f"\n{'='*80}")
    print(f"  PICK UNIVERSE — {datetime.now().strftime('%B %d, %Y')}")
    print(f"  Advanced Stats × BallparkPal Simulation")
    print(f"{'='*80}")

    labels = {
        'ELITE': 'ELITE — High HR + Low K% + Environment + Grade all confirmed',
        'VALIDATED': 'VALIDATED — Matchup + Environment + Grade confirmed',
        'MATCHUP_ONLY': 'MATCHUP ONLY — Great raw matchup, park not helping',
        'SUPPRESSED': 'SUPPRESSED — Elite matchup killed by park (series watch)',
    }
    for tier in ['ELITE', 'VALIDATED', 'MATCHUP_ONLY', 'SUPPRESSED']:
        tier_picks = picks_df[picks_df['tier'] == tier]
        if len(tier_picks) == 0:
            continue
        print(f"\n--- {labels[tier]} ---")
        for _, p in tier_picks.head(top_n).iterrows():
            bvp = f"BvP:{int(p.get('PA',0))}PA/{int(p.get('HR (hist)',0))}HR" if p.get('PA', 0) > 0 else ""
            print(f"  {p['Batter']:22s} {p['Team']:4s} vs {p['Pitcher']:12s} "
                  f"| HR:{p['HR Prob']:.1f}%(raw:{p['HR Prob (no park)']:.1f}%) "
                  f"| Bst:{p['HR Boost']:+d}% | K:{p['K Prob']:.0f}% "
                  f"| vs:{p['vs Grade']:+d} | RC:{p['RC']:+d} {bvp}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ballparkpal_integration.py <Matchups_YYYY-MM-DD.xlsx>")
    else:
        picks = generate_pick_universe(sys.argv[1])
        print_pick_universe(picks)
