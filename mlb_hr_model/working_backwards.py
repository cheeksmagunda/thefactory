"""
MLB HR Model — Working Backwards Log
======================================

Track and analyze winning HR picks from sharp bettors/models
to reverse-engineer their methodology and apply it to ours.

Usage:
    python working_backwards.py add       # Add a new pick to the log
    python working_backwards.py analyze   # Analyze patterns in the log
    python working_backwards.py show      # Display all logged picks
"""

import json
import os
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), 'data', 'working_backwards_log.json')


def load_log():
    """Load the working backwards log."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {'picks': [], 'patterns': [], 'last_updated': None}


def save_log(log):
    """Save the log."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    log['last_updated'] = datetime.now().isoformat()
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


def add_pick(source, date, batter_name, batter_team, pitcher_name, pitcher_team,
             odds, wager=None, payout=None, won=True, venue=None,
             temperature=None, wind=None, notes=None, 
             parlay_legs=None, boost_pct=None):
    """
    Add a sharp bettor's pick to the log for analysis.
    
    Args:
        source: Who made the pick (e.g., "GiuseppePaps", "HomeRunPredict")
        date: Game date (YYYY-MM-DD)
        batter_name: Hitter name
        batter_team: Hitter's team
        pitcher_name: Opposing pitcher
        pitcher_team: Pitcher's team
        odds: American odds on the HR prop
        wager: Amount wagered (optional)
        payout: Amount paid out (optional)
        won: Whether the pick hit (default True — we're logging winners)
        venue: Ballpark
        temperature: Game temp
        wind: Wind description
        notes: Any additional context
        parlay_legs: List of other legs if it was a parlay
        boost_pct: Profit boost % if used
    """
    log = load_log()
    
    pick = {
        'id': len(log['picks']) + 1,
        'source': source,
        'date': date,
        'batter_name': batter_name,
        'batter_team': batter_team,
        'pitcher_name': pitcher_name,
        'pitcher_team': pitcher_team,
        'odds': odds,
        'wager': wager,
        'payout': payout,
        'won': won,
        'venue': venue,
        'temperature': temperature,
        'wind': wind,
        'notes': notes,
        'parlay_legs': parlay_legs,
        'boost_pct': boost_pct,
        'logged_at': datetime.now().isoformat(),
    }
    
    log['picks'].append(pick)
    save_log(log)
    print(f"Logged: {batter_name} ({batter_team}) vs {pitcher_name} at {odds} — {source} [{date}]")
    return pick


def analyze_patterns():
    """Analyze the log for recurring patterns."""
    log = load_log()
    picks = log['picks']
    
    if not picks:
        print("No picks logged yet.")
        return
    
    print(f"\n{'='*60}")
    print(f"WORKING BACKWARDS ANALYSIS — {len(picks)} picks logged")
    print(f"{'='*60}")
    
    # Sources
    sources = {}
    for p in picks:
        src = p['source']
        if src not in sources:
            sources[src] = {'total': 0, 'won': 0, 'odds_sum': 0}
        sources[src]['total'] += 1
        sources[src]['won'] += 1 if p['won'] else 0
        sources[src]['odds_sum'] += p.get('odds', 0)
    
    print(f"\nBy source:")
    for src, data in sources.items():
        avg_odds = data['odds_sum'] / data['total']
        print(f"  {src}: {data['total']} picks, {data['won']} won, avg odds +{avg_odds:.0f}")
    
    # Odds distribution
    odds_list = [p['odds'] for p in picks if p.get('odds')]
    if odds_list:
        print(f"\nOdds distribution:")
        print(f"  Average: +{sum(odds_list)/len(odds_list):.0f}")
        print(f"  Range: +{min(odds_list)} to +{max(odds_list)}")
        
        # Buckets
        buckets = {'<+300': 0, '+300-499': 0, '+500-699': 0, '+700+': 0}
        for o in odds_list:
            if o < 300: buckets['<+300'] += 1
            elif o < 500: buckets['+300-499'] += 1
            elif o < 700: buckets['+500-699'] += 1
            else: buckets['+700+'] += 1
        for bucket, count in buckets.items():
            bar = '█' * count
            print(f"  {bucket:10s}: {bar} ({count})")
    
    # Pitcher analysis
    pitchers_targeted = {}
    for p in picks:
        pit = p.get('pitcher_name', 'Unknown')
        if pit not in pitchers_targeted:
            pitchers_targeted[pit] = 0
        pitchers_targeted[pit] += 1
    
    repeat_pitchers = {k: v for k, v in pitchers_targeted.items() if v > 1}
    if repeat_pitchers:
        print(f"\nRepeat pitcher targets (attacked multiple times):")
        for pit, count in sorted(repeat_pitchers.items(), key=lambda x: -x[1]):
            print(f"  {pit}: {count} times")
    
    # Batter analysis
    batters_picked = {}
    for p in picks:
        bat = p.get('batter_name', 'Unknown')
        if bat not in batters_picked:
            batters_picked[bat] = 0
        batters_picked[bat] += 1
    
    repeat_batters = {k: v for k, v in batters_picked.items() if v > 1}
    if repeat_batters:
        print(f"\nRepeat batter picks:")
        for bat, count in sorted(repeat_batters.items(), key=lambda x: -x[1]):
            print(f"  {bat}: {count} times")
    
    # Series persistence
    date_team_combos = {}
    for p in picks:
        key = f"{p['batter_team']} vs {p['pitcher_team']}"
        if key not in date_team_combos:
            date_team_combos[key] = []
        date_team_combos[key].append(p['date'])
    
    series_plays = {k: v for k, v in date_team_combos.items() if len(v) > 1}
    if series_plays:
        print(f"\nSeries persistence (same matchup hit multiple days):")
        for matchup, dates in series_plays.items():
            print(f"  {matchup}: {', '.join(sorted(dates))}")
    
    # SGP / parlay analysis
    parlays = [p for p in picks if p.get('parlay_legs')]
    if parlays:
        print(f"\nParlay structure:")
        print(f"  {len(parlays)} of {len(picks)} picks were parlays/SGPs")
        same_team_stacks = [p for p in parlays if any(
            leg.get('team') == p['batter_team'] for leg in (p['parlay_legs'] or [])
        )]
        print(f"  Same-team HR stacks: {len(same_team_stacks)}")
    
    # Key patterns summary
    print(f"\n{'='*60}")
    print(f"KEY PATTERNS IDENTIFIED:")
    print(f"{'='*60}")
    
    patterns = []
    
    if odds_list and sum(o >= 400 for o in odds_list) / len(odds_list) > 0.5:
        patterns.append("Deep value preferred — majority of picks at +400 or longer")
    
    if repeat_pitchers:
        patterns.append(f"Pitcher targeting — same vulnerable arms attacked multiple times")
    
    if series_plays:
        patterns.append(f"Series persistence — profitable matchups pressed across multiple days")
    
    if repeat_batters:
        patterns.append(f"Hot streak exploitation — same hitters picked repeatedly")
    
    boost_picks = [p for p in picks if p.get('boost_pct')]
    if boost_picks:
        patterns.append(f"Profit boost amplification — {len(boost_picks)} picks used boosts")
    
    for i, pat in enumerate(patterns, 1):
        print(f"  {i}. {pat}")
    
    log['patterns'] = patterns
    save_log(log)
    
    return patterns


def show_log():
    """Display all logged picks."""
    log = load_log()
    picks = log['picks']
    
    if not picks:
        print("No picks logged yet. Use 'add' to log picks.")
        return
    
    print(f"\n{'='*70}")
    print(f"WORKING BACKWARDS LOG — {len(picks)} picks")
    print(f"{'='*70}")
    
    for p in picks:
        won_str = "W" if p['won'] else "L"
        odds_str = f"+{p['odds']}" if p['odds'] > 0 else str(p['odds'])
        parlay_str = f" (SGP {len(p['parlay_legs'])+1} legs)" if p.get('parlay_legs') else ""
        boost_str = f" [{p['boost_pct']}% boost]" if p.get('boost_pct') else ""
        
        print(f"\n  [{won_str}] {p['date']} | {p['source']}")
        print(f"      {p['batter_name']} ({p['batter_team']}) vs {p['pitcher_name']} ({p['pitcher_team']})")
        print(f"      Odds: {odds_str}{parlay_str}{boost_str}")
        if p.get('wager') and p.get('payout'):
            print(f"      ${p['wager']} → ${p['payout']}")
        if p.get('notes'):
            print(f"      Notes: {p['notes']}")


# ============================================================================
# SEED WITH GIUSEPPE'S PICKS
# ============================================================================

def seed_giuseppe_picks():
    """Pre-load Giuseppe's recent winning picks from our analysis."""
    
    picks_to_add = [
        {
            'source': 'GiuseppePaps',
            'date': '2026-03-31',
            'batter_name': 'Pete Alonso',
            'batter_team': 'BAL',
            'pitcher_name': 'Jacob deGrom',
            'pitcher_team': 'TEX',
            'odds': 600,
            'wager': 50,
            'payout': 1550,
            'venue': 'BAL',
            'notes': 'SGP with Danny Jansen. Eflin pitching (5.93 ERA, 2.27 HR/9). 80F at Camden.',
            'parlay_legs': [{'name': 'Danny Jansen HR', 'team': 'TEX', 'odds': 500}],
        },
        {
            'source': 'GiuseppePaps',
            'date': '2026-04-01',
            'batter_name': 'Corey Seager',
            'batter_team': 'TEX',
            'pitcher_name': 'BAL starter',
            'pitcher_team': 'BAL',
            'odds': 505,
            'wager': 750,
            'payout': 4537,
            'venue': 'BAL',
            'notes': 'Single HR prop. Day 2 of TEX@BAL series — pressed the matchup after day 1 hit.',
        },
        {
            'source': 'GiuseppePaps',
            'date': '2026-04-06',
            'batter_name': 'CJ Abrams',
            'batter_team': 'WSH',
            'pitcher_name': 'M. Liberatore',
            'pitcher_team': 'STL',
            'odds': 800,
            'wager': 25,
            'payout': 4413,
            'venue': 'WSH',
            'notes': 'SGP+ with James Wood HR + run totals. 70% profit boost. Same-team HR stack.',
            'parlay_legs': [
                {'name': 'James Wood HR', 'team': 'WSH'},
                {'name': 'WSH over 1.5 runs', 'type': 'filler'},
                {'name': 'LAD over 1.5 runs', 'type': 'filler'},
            ],
            'boost_pct': 70,
        },
        {
            'source': 'GiuseppePaps',
            'date': '2026-04-06',
            'batter_name': 'Drake Baldwin',
            'batter_team': 'ATL',
            'pitcher_name': 'Jose Soriano',
            'pitcher_team': 'LAA',
            'odds': 680,
            'wager': 441,
            'payout': 3439,
            'venue': 'ATL',
            'notes': 'Single HR prop. Baldwin on a heater — 5 HR in 13 games. Books slow to adjust.',
        },
        {
            'source': 'GiuseppePaps',
            'date': '2026-04-07',
            'batter_name': 'James Wood',
            'batter_team': 'WSH',
            'pitcher_name': 'M. Liberatore',
            'pitcher_team': 'STL',
            'odds': 500,
            'wager': 500,
            'payout': 5700,
            'venue': 'WSH',
            'notes': 'SGP with K filler legs. Day 2 targeting Wood vs STL. 30% profit boost.',
            'parlay_legs': [
                {'name': 'Cade Cavalli 1+ K', 'type': 'filler'},
                {'name': 'M. Liberatore 1+ K', 'type': 'filler'},
            ],
            'boost_pct': 30,
        },
        {
            'source': 'GiuseppePaps',
            'date': '2026-04-07',
            'batter_name': 'Jorge Soler',
            'batter_team': 'LAA',
            'pitcher_name': 'ATL starter',
            'pitcher_team': 'ATL',
            'odds': 400,
            'wager': 275,
            'payout': 1718,
            'venue': 'ATL',
            'notes': 'SGP with Kikuchi K filler. Raw power bat at plus money.',
            'parlay_legs': [
                {'name': 'Yusei Kikuchi 1+ K', 'type': 'filler'},
            ],
        },
    ]
    
    log = load_log()
    
    # Only add if log is empty (avoid duplicates)
    if log['picks']:
        print(f"Log already has {len(log['picks'])} picks. Skipping seed.")
        return
    
    for pick_data in picks_to_add:
        add_pick(**pick_data)
    
    print(f"\nSeeded {len(picks_to_add)} Giuseppe picks into the log.")


# ============================================================================
# CLI
# ============================================================================

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python working_backwards.py [add|analyze|show|seed]")
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == 'seed':
        seed_giuseppe_picks()
    elif cmd == 'show':
        show_log()
    elif cmd == 'analyze':
        analyze_patterns()
    elif cmd == 'add':
        # Interactive add
        print("Add a sharp pick to the log:")
        source = input("  Source (e.g., GiuseppePaps): ").strip()
        date = input("  Date (YYYY-MM-DD): ").strip()
        batter = input("  Batter name: ").strip()
        b_team = input("  Batter team: ").strip()
        pitcher = input("  Pitcher name: ").strip()
        p_team = input("  Pitcher team: ").strip()
        odds = int(input("  Odds (e.g., 500): ").strip())
        notes = input("  Notes (optional): ").strip() or None
        
        add_pick(source, date, batter, b_team, pitcher, p_team, odds, notes=notes)
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
