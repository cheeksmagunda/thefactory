"""
MLB HR Model — Daily Picks Pipeline
=====================================

Run this each day to generate HR pick recommendations.

Usage:
    python daily_picks.py                    # Interactive mode — prompts for slate
    python daily_picks.py --slate today.csv  # Load slate from CSV
    python daily_picks.py --demo             # Demo with sample data

Slate CSV format:
    batter_name, batter_id, pitcher_name, pitcher_id, team, opp_team, venue,
    stand, p_throws, odds, temperature, wind_speed, wind_dir

The model evaluates each batter using:
1. Power conviction — can this hitter actually go yard? (barrel rate, ISO, EV, form)
2. Market mispricing — is the book offering value? (model prob vs implied prob)
3. Environment boost — park, weather, platoon working in his favor?

"Triple convergence" = all three layers align. That's where we bet.
"""

import pandas as pd
import numpy as np
import os
import json
import sys
from datetime import datetime

# ============================================================================
# ODDS CONVERSION UTILITIES
# ============================================================================

def american_to_implied(odds):
    """Convert American odds to implied probability (with vig)."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def american_to_implied_novig(odds, vig_pct=0.05):
    """Convert American odds to vig-removed implied probability."""
    raw = american_to_implied(odds)
    return raw * (1 - vig_pct)

def implied_to_american(prob):
    """Convert probability to American odds."""
    if prob <= 0 or prob >= 1:
        return 0
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    else:
        return round(100 * (1 - prob) / prob)


# ============================================================================
# PLAYER DATABASE — Season stats lookup
# ============================================================================

def load_player_database(parquet_path=None):
    """
    Load the most recent season stats for all batters and pitchers
    from our training data.
    
    Returns two dicts: batter_stats[player_id] and pitcher_stats[player_id]
    """
    if parquet_path is None:
        # Try to find cached data
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        candidates = [f for f in os.listdir(data_dir) if f.endswith('.parquet')] if os.path.exists(data_dir) else []
        if not candidates:
            print("No cached Statcast data found. Run the training pipeline first.")
            return None, None, {}
        parquet_path = os.path.join(data_dir, sorted(candidates)[-1])
    
    print(f"Loading player database from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Filter to most recent season for freshest stats
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['season'] = df['game_date'].dt.year
    latest_season = df['season'].max()
    recent = df[df['season'] == latest_season].copy()
    
    # Filter to PA-ending events
    pa = recent[recent['events'].notna()].copy()
    
    # Compute batted ball indicators
    ev = pa['launch_speed'].fillna(0)
    la = pa['launch_angle'].fillna(0)
    pa['is_hr'] = (pa['events'] == 'home_run').astype(int)
    pa['is_barrel'] = ((ev >= 98) & (la.between(26, 30)) |
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
    pa['is_fly_ball'] = la.between(25, 60).astype(int)
    pa['is_hard_hit'] = (ev >= 95).astype(int)
    
    # Batter season stats
    batter_stats = pa.groupby('batter').agg(
        pa_count=('is_hr', 'count'),
        hr_count=('is_hr', 'sum'),
        hr_rate=('is_hr', 'mean'),
        barrel_rate=('is_barrel', 'mean'),
        avg_ev=('launch_speed', lambda x: x.dropna().mean()),
        avg_la=('launch_angle', lambda x: x.dropna().mean()),
        fb_rate=('is_fly_ball', 'mean'),
        hard_hit_rate=('is_hard_hit', 'mean'),
    ).to_dict('index')
    
    # Recent form (last 60 PAs)
    pa_sorted = pa.sort_values(['batter', 'game_date', 'at_bat_number'])
    recent_form = pa_sorted.groupby('batter').tail(60).groupby('batter').agg(
        hr_rate_recent=('is_hr', 'mean'),
        barrel_rate_recent=('is_barrel', 'mean'),
        ev_recent=('launch_speed', lambda x: x.dropna().mean()),
    ).to_dict('index')
    
    # Merge recent form into batter stats
    for bid in batter_stats:
        if bid in recent_form:
            batter_stats[bid]['hr_rate_recent'] = recent_form[bid]['hr_rate_recent']
            batter_stats[bid]['hr_form_delta'] = (
                recent_form[bid]['hr_rate_recent'] - batter_stats[bid]['hr_rate']
            )
        else:
            batter_stats[bid]['hr_rate_recent'] = batter_stats[bid]['hr_rate']
            batter_stats[bid]['hr_form_delta'] = 0
    
    # Pitcher season stats
    pitcher_stats = pa.groupby('pitcher').agg(
        pa_faced=('is_hr', 'count'),
        hr_allowed=('is_hr', 'sum'),
        hr_rate_allowed=('is_hr', 'mean'),
        barrel_rate_allowed=('is_barrel', 'mean'),
        avg_ev_allowed=('launch_speed', lambda x: x.dropna().mean()),
        fb_rate_allowed=('is_fly_ball', 'mean'),
        hard_hit_rate_allowed=('is_hard_hit', 'mean'),
    ).to_dict('index')
    
    # Batter name lookup
    if 'player_name' in pa.columns:
        name_map = pa.groupby('batter')['player_name'].first().to_dict()
    else:
        name_map = {}
    
    print(f"Loaded {len(batter_stats)} batters, {len(pitcher_stats)} pitchers ({latest_season} season)")
    
    return batter_stats, pitcher_stats, name_map


# ============================================================================
# POWER SCORE — Can this hitter actually go yard?
# ============================================================================

def compute_power_score(batter_id, batter_stats):
    """
    Score 0-100 for raw HR power ability.
    Based on our model's top features: barrel rate, ISO proxy, EV, form.
    """
    if batter_id not in batter_stats:
        return 30  # Unknown hitter — default to low
    
    s = batter_stats[batter_id]
    
    # Weighted components (aligned with our feature importance)
    barrel_score = min(s.get('barrel_rate', 0) / 0.15 * 100, 100)  # 15% barrel = 100
    ev_score = min(max(s.get('avg_ev', 85) - 85, 0) / 8 * 100, 100)  # 93+ mph = 100
    hr_rate_score = min(s.get('hr_rate', 0) / 0.06 * 100, 100)  # 6% HR rate = 100
    form_bonus = max(s.get('hr_form_delta', 0) * 500, -20)  # Hot streak bonus
    fb_score = min(s.get('fb_rate', 0) / 0.45 * 100, 100)  # 45% FB rate = 100
    
    power = (
        barrel_score * 0.30 +
        ev_score * 0.20 +
        hr_rate_score * 0.25 +
        fb_score * 0.10 +
        min(max(form_bonus, 0), 15) * 1.0  # Up to 15 point bonus for hot streak
    )
    
    return min(round(power, 1), 100)


# ============================================================================
# PITCHER VULNERABILITY — How hittable is this pitcher for HRs?
# ============================================================================

def compute_pitcher_vulnerability(pitcher_id, pitcher_stats):
    """
    Score 0-100 for how vulnerable this pitcher is to giving up HRs.
    Higher = more HR-prone. Our model's #1 feature is pitcher FB velo (low = bad).
    """
    if pitcher_id not in pitcher_stats:
        return 50  # Unknown pitcher — default to average
    
    s = pitcher_stats[pitcher_id]
    
    hr_rate_score = min(s.get('hr_rate_allowed', 0.03) / 0.06 * 100, 100)
    barrel_score = min(s.get('barrel_rate_allowed', 0.06) / 0.12 * 100, 100)
    ev_score = min(max(s.get('avg_ev_allowed', 87) - 85, 0) / 6 * 100, 100)
    hard_hit_score = min(s.get('hard_hit_rate_allowed', 0.35) / 0.45 * 100, 100)
    fb_score = min(s.get('fb_rate_allowed', 0.25) / 0.40 * 100, 100)
    
    vuln = (
        hr_rate_score * 0.30 +
        barrel_score * 0.20 +
        ev_score * 0.15 +
        hard_hit_score * 0.20 +
        fb_score * 0.15
    )
    
    return min(round(vuln, 1), 100)


# ============================================================================
# ENVIRONMENT SCORE — Weather, park, platoon helping?
# ============================================================================

def compute_environment_score(temperature=None, wind_speed=None, wind_dir=None,
                               platoon_advantage=False, venue=None):
    """
    Score 0-100 for how favorable the environment is for HRs.
    Based on BallparkPal methodology insights.
    """
    score = 50  # Neutral baseline
    
    # Temperature: +1 point per degree above 65, -1 per degree below
    if temperature is not None:
        score += (temperature - 65) * 0.8
    
    # Wind: blowing out = good, in = bad
    if wind_speed is not None and wind_dir is not None:
        wind_dir = wind_dir.lower()
        if 'out' in wind_dir:
            score += wind_speed * 1.2
        elif 'in' in wind_dir:
            score -= wind_speed * 1.0
        # L-R or R-L winds have moderate effect
        elif 'l' in wind_dir or 'r' in wind_dir:
            score += wind_speed * 0.3
    
    # Platoon advantage: ~15% more HRs historically
    if platoon_advantage:
        score += 8
    
    # Known HR-friendly parks
    hr_parks = {
        'CIN': 12, 'COL': 15, 'PHI': 10, 'NYY': 8, 'CHC': 5,
        'MIL': 5, 'ATL': 4, 'BAL': 4, 'MIN': 3, 'TEX': 3,
    }
    # HR-suppressing parks  
    pitcher_parks = {
        'SF': -10, 'SEA': -6, 'MIA': -5, 'OAK': -5, 'SD': -4,
        'STL': -3, 'KC': -2, 'DET': -2,
    }
    if venue:
        score += hr_parks.get(venue, 0) + pitcher_parks.get(venue, 0)
    
    return min(max(round(score, 1), 0), 100)


# ============================================================================
# MODEL HR PROBABILITY ESTIMATE 
# ============================================================================

def estimate_hr_probability(power_score, pitcher_vuln, env_score, 
                            batter_stats_dict=None, expected_pa=4.0):
    """
    Estimate game-level HR probability from our three convergence scores.
    
    This is a simplified proxy used when the full LightGBM model isn't loaded.
    Maps the composite score to a probability calibrated against historical rates.
    
    Base HR rate per PA: ~3.1%
    Base HR rate per game: ~11.5%
    """
    # Composite score (0-100)
    composite = (power_score * 0.40 + pitcher_vuln * 0.35 + env_score * 0.25)
    
    # Map composite to per-PA HR probability
    # 50 composite → ~3.1% (league average)
    # 80 composite → ~6.5% (elite matchup)  
    # 30 composite → ~1.5% (poor matchup)
    pa_prob = 0.031 * (composite / 50) ** 1.3
    pa_prob = min(max(pa_prob, 0.005), 0.12)  # Floor/cap
    
    # Game-level: P(≥1 HR) = 1 - (1 - pa_prob)^expected_pa
    game_prob = 1 - (1 - pa_prob) ** expected_pa
    
    return round(game_prob, 4), round(pa_prob, 4)


# ============================================================================
# EDGE DETECTION + TRIPLE CONVERGENCE CLASSIFICATION
# ============================================================================

def classify_pick(power_score, pitcher_vuln, env_score, edge_pct):
    """
    Classify a pick into our framework tiers.
    
    Triple Convergence: power ≥ 55, pitcher_vuln ≥ 55, env ≥ 55, edge ≥ 3%
    Value + Power: power ≥ 45, edge ≥ 3%
    Power Play: power ≥ 60, edge ≥ 1%
    Monitor: anything with edge ≥ 0%
    Skip: negative edge or low power
    """
    if power_score >= 55 and pitcher_vuln >= 55 and env_score >= 55 and edge_pct >= 3.0:
        return "TRIPLE CONVERGENCE"
    elif power_score >= 45 and edge_pct >= 3.0:
        return "VALUE + POWER"
    elif power_score >= 60 and edge_pct >= 1.0:
        return "POWER PLAY"
    elif edge_pct >= 0:
        return "MONITOR"
    else:
        return "SKIP"


# ============================================================================
# PAIRING ENGINE — Best 2-leg HR combos
# ============================================================================

def generate_pairings(picks_df, top_n=5):
    """
    Generate the best HR parlay pairings from tonight's picks.
    
    Prioritizes:
    1. Both legs individually +EV
    2. Independent games preferred (but same-game correlation noted)
    3. Combined edge maximized
    """
    # Only pair picks with positive edge
    eligible = picks_df[picks_df['edge_pct'] > 0].copy()
    
    if len(eligible) < 2:
        print("Not enough +EV picks for pairings")
        return pd.DataFrame()
    
    pairings = []
    
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            a = eligible.iloc[i]
            b = eligible.iloc[j]
            
            same_game = (a.get('game_id', '') == b.get('game_id', '')) if 'game_id' in eligible.columns else (a.get('venue', '') == b.get('venue', ''))
            same_team = (a.get('team', '') == b.get('team', ''))
            
            # Combined probability (adjust for correlation if same game)
            corr_factor = 1.15 if same_game else 1.0  # Same game = slight positive correlation
            combined_prob = a['model_prob'] * b['model_prob'] * corr_factor
            
            # Combined book implied
            dec_a = (a['odds'] / 100 + 1) if a['odds'] > 0 else (100 / abs(a['odds']) + 1)
            dec_b = (b['odds'] / 100 + 1) if b['odds'] > 0 else (100 / abs(b['odds']) + 1)
            parlay_decimal = dec_a * dec_b
            parlay_american = round((parlay_decimal - 1) * 100) if parlay_decimal >= 2 else round(-100 / (parlay_decimal - 1))
            book_combined = 1 / parlay_decimal
            
            combined_edge = combined_prob - book_combined
            
            pairings.append({
                'player_a': a.get('batter_name', a.get('batter_id', '?')),
                'player_b': b.get('batter_name', b.get('batter_id', '?')),
                'odds_a': a['odds'],
                'odds_b': b['odds'],
                'parlay_odds': parlay_american,
                'model_combined_prob': round(combined_prob * 100, 2),
                'book_combined_prob': round(book_combined * 100, 2),
                'combined_edge_pct': round(combined_edge * 100, 2),
                'same_game': same_game,
                'same_team': same_team,
                'type': 'Same-team stack' if same_team else ('Same-game' if same_game else 'Cross-game'),
            })
    
    pairs_df = pd.DataFrame(pairings).sort_values('combined_edge_pct', ascending=False)
    return pairs_df.head(top_n)


# ============================================================================
# SLATE INPUT — Manual entry for today's matchups
# ============================================================================

def input_slate_manually():
    """Interactive prompt for entering today's slate."""
    print("\n" + "=" * 60)
    print("Enter today's HR candidates")
    print("=" * 60)
    print("For each hitter, enter: name, team, pitcher_name, opp_team, odds, temp, wind, stand, p_throws")
    print("Type 'done' when finished.\n")
    
    entries = []
    while True:
        line = input(">>> ").strip()
        if line.lower() == 'done':
            break
        
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 5:
            print("  Need at least: name, team, pitcher, opp_team, odds")
            continue
        
        entry = {
            'batter_name': parts[0],
            'team': parts[1],
            'pitcher_name': parts[2],
            'opp_team': parts[3],
            'odds': int(parts[4]),
            'temperature': float(parts[5]) if len(parts) > 5 and parts[5] else None,
            'wind_speed': float(parts[6]) if len(parts) > 6 and parts[6] else None,
            'wind_dir': parts[7] if len(parts) > 7 else None,
            'stand': parts[8] if len(parts) > 8 else None,
            'p_throws': parts[9] if len(parts) > 9 else None,
        }
        entries.append(entry)
        print(f"  Added: {entry['batter_name']} ({entry['team']}) vs {entry['pitcher_name']} at {entry['odds']}")
    
    return pd.DataFrame(entries)


def load_slate_csv(filepath):
    """Load slate from a CSV file."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} entries from {filepath}")
    return df


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_slate(slate_df, batter_stats, pitcher_stats, name_map=None):
    """
    Run the full analysis on a slate of hitters.
    Returns ranked picks with scores, probabilities, edges, and classifications.
    """
    results = []
    
    for _, row in slate_df.iterrows():
        batter_name = row.get('batter_name', '?')
        batter_id = row.get('batter_id', None)
        pitcher_name = row.get('pitcher_name', '?')
        pitcher_id = row.get('pitcher_id', None)
        odds = row.get('odds', 400)
        
        # If we have IDs, use them; otherwise try name matching
        # (In production, we'd have a proper name→ID resolver)
        
        # Compute scores
        power = compute_power_score(batter_id, batter_stats) if batter_id else 50
        vuln = compute_pitcher_vulnerability(pitcher_id, pitcher_stats) if pitcher_id else 50
        
        platoon = False
        if row.get('stand') and row.get('p_throws'):
            platoon = (row['stand'] != row['p_throws'])
        
        env = compute_environment_score(
            temperature=row.get('temperature'),
            wind_speed=row.get('wind_speed'),
            wind_dir=row.get('wind_dir'),
            platoon_advantage=platoon,
            venue=row.get('venue') or row.get('opp_team')
        )
        
        # HR probability
        game_prob, pa_prob = estimate_hr_probability(power, vuln, env)
        
        # Edge vs market
        market_implied = american_to_implied_novig(odds)
        edge = game_prob - market_implied
        edge_pct = edge * 100
        
        # Classification
        tier = classify_pick(power, vuln, env, edge_pct)
        
        results.append({
            'batter_name': batter_name,
            'batter_id': batter_id,
            'team': row.get('team', '?'),
            'pitcher_name': pitcher_name,
            'opp_team': row.get('opp_team', '?'),
            'odds': odds,
            'power_score': power,
            'pitcher_vuln': vuln,
            'env_score': env,
            'model_prob': round(game_prob, 4),
            'market_implied': round(market_implied, 4),
            'edge_pct': round(edge_pct, 2),
            'tier': tier,
            'venue': row.get('venue', row.get('opp_team', '?')),
        })
    
    results_df = pd.DataFrame(results).sort_values('edge_pct', ascending=False)
    return results_df


def print_picks(results_df):
    """Pretty-print the pick recommendations."""
    print("\n" + "=" * 70)
    print(f"  HR PICKS — {datetime.now().strftime('%B %d, %Y')}")
    print("=" * 70)
    
    for tier_name in ["TRIPLE CONVERGENCE", "VALUE + POWER", "POWER PLAY", "MONITOR"]:
        tier_picks = results_df[results_df['tier'] == tier_name]
        if len(tier_picks) == 0:
            continue
        
        print(f"\n--- {tier_name} ---")
        for _, p in tier_picks.iterrows():
            edge_str = f"+{p['edge_pct']:.1f}%" if p['edge_pct'] > 0 else f"{p['edge_pct']:.1f}%"
            print(f"  {p['batter_name']:20s} {p['team']:4s} vs {p['pitcher_name']:18s} "
                  f"| {'+' if p['odds'] > 0 else ''}{p['odds']} "
                  f"| Model: {p['model_prob']*100:.1f}% "
                  f"| Edge: {edge_str} "
                  f"| Pwr:{p['power_score']:.0f} Vuln:{p['pitcher_vuln']:.0f} Env:{p['env_score']:.0f}")
    
    skips = results_df[results_df['tier'] == 'SKIP']
    if len(skips) > 0:
        print(f"\n--- SKIP ({len(skips)} hitters) ---")
        for _, p in skips.head(3).iterrows():
            print(f"  {p['batter_name']:20s} | {'+' if p['odds'] > 0 else ''}{p['odds']} "
                  f"| Edge: {p['edge_pct']:.1f}% — overpriced")


def print_pairings(pairs_df):
    """Pretty-print the pairing recommendations."""
    if len(pairs_df) == 0:
        return
    
    print("\n" + "=" * 70)
    print("  BEST PAIRINGS")
    print("=" * 70)
    
    for i, (_, p) in enumerate(pairs_df.iterrows()):
        tag = f"[{p['type']}]"
        print(f"\n  #{i+1} {p['player_a']} + {p['player_b']}  {tag}")
        print(f"     Parlay: {'+' if p['parlay_odds'] > 0 else ''}{p['parlay_odds']} "
              f"| Model: {p['model_combined_prob']:.1f}% "
              f"| Book: {p['book_combined_prob']:.1f}% "
              f"| Edge: +{p['combined_edge_pct']:.1f}%")


# ============================================================================
# DEMO DATA — Sample slate for testing
# ============================================================================

def get_demo_slate():
    """Demo slate based on April 10, 2026 analysis."""
    return pd.DataFrame([
        {'batter_name': 'Kyle Schwarber', 'team': 'PHI', 'pitcher_name': 'M. Soroka', 'opp_team': 'ARI', 'odds': 270, 'temperature': 68, 'wind_speed': 8, 'wind_dir': 'out', 'stand': 'L', 'p_throws': 'R', 'venue': 'PHI'},
        {'batter_name': 'Matt Olson', 'team': 'ATL', 'pitcher_name': 'S. Cecconi', 'opp_team': 'CLE', 'odds': 350, 'temperature': 76, 'wind_speed': 2, 'wind_dir': 'L-R', 'stand': 'L', 'p_throws': 'R', 'venue': 'ATL'},
        {'batter_name': 'Drake Baldwin', 'team': 'ATL', 'pitcher_name': 'S. Cecconi', 'opp_team': 'CLE', 'odds': 490, 'temperature': 76, 'wind_speed': 2, 'wind_dir': 'L-R', 'stand': 'L', 'p_throws': 'R', 'venue': 'ATL'},
        {'batter_name': 'Kerry Carpenter', 'team': 'DET', 'pitcher_name': 'C. Paddack', 'opp_team': 'MIA', 'odds': 320, 'temperature': 49, 'wind_speed': 9, 'wind_dir': 'out', 'stand': 'L', 'p_throws': 'R', 'venue': 'MIA'},
        {'batter_name': 'Bryce Harper', 'team': 'PHI', 'pitcher_name': 'M. Soroka', 'opp_team': 'ARI', 'odds': 360, 'temperature': 68, 'wind_speed': 8, 'wind_dir': 'out', 'stand': 'L', 'p_throws': 'R', 'venue': 'PHI'},
        {'batter_name': 'Aaron Judge', 'team': 'NYY', 'pitcher_name': 'S. Matz', 'opp_team': 'TB', 'odds': 300, 'temperature': 72, 'wind_speed': 0, 'wind_dir': 'dome', 'stand': 'R', 'p_throws': 'L', 'venue': 'TB'},
        {'batter_name': 'Kyle Tucker', 'team': 'LAD', 'pitcher_name': 'K. Rocker', 'opp_team': 'TEX', 'odds': 360, 'temperature': 66, 'wind_speed': 11, 'wind_dir': 'out', 'stand': 'L', 'p_throws': 'R', 'venue': 'LAD'},
        {'batter_name': 'Shohei Ohtani', 'team': 'LAD', 'pitcher_name': 'K. Rocker', 'opp_team': 'TEX', 'odds': 180, 'temperature': 66, 'wind_speed': 11, 'wind_dir': 'out', 'stand': 'L', 'p_throws': 'R', 'venue': 'LAD'},
        {'batter_name': 'Fernando Tatis', 'team': 'SD', 'pitcher_name': 'T. Sugano', 'opp_team': 'COL', 'odds': 390, 'temperature': 67, 'wind_speed': 9, 'wind_dir': 'L-R', 'stand': 'R', 'p_throws': 'R', 'venue': 'SD'},
        {'batter_name': 'Corey Seager', 'team': 'TEX', 'pitcher_name': 'T. Glasnow', 'opp_team': 'LAD', 'odds': 410, 'temperature': 66, 'wind_speed': 11, 'wind_dir': 'out', 'stand': 'L', 'p_throws': 'R', 'venue': 'LAD'},
    ])


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='MLB HR Daily Picks')
    parser.add_argument('--slate', type=str, help='Path to slate CSV')
    parser.add_argument('--demo', action='store_true', help='Run with demo data')
    parser.add_argument('--interactive', action='store_true', help='Enter slate manually')
    args = parser.parse_args()
    
    # Load player database
    batter_stats, pitcher_stats, name_map = load_player_database()
    
    # Get slate
    if args.demo:
        slate = get_demo_slate()
        print("Running demo with April 10 slate...")
    elif args.slate:
        slate = load_slate_csv(args.slate)
    elif args.interactive:
        slate = input_slate_manually()
    else:
        # Default to demo
        print("No slate specified. Use --demo, --slate <file>, or --interactive")
        print("Running demo mode...\n")
        slate = get_demo_slate()
    
    # Analyze
    results = analyze_slate(slate, batter_stats or {}, pitcher_stats or {})
    
    # Output picks
    print_picks(results)
    
    # Generate pairings
    pairs = generate_pairings(results)
    print_pairings(pairs)
    
    # Save results
    os.makedirs('output', exist_ok=True)
    date_str = datetime.now().strftime('%Y-%m-%d')
    results.to_csv(f'output/picks_{date_str}.csv', index=False)
    print(f"\nPicks saved to output/picks_{date_str}.csv")
    
    return results, pairs


if __name__ == "__main__":
    main()
