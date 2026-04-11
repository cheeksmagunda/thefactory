"""
MLB HR Model — Advanced Feature Engineering (Statcast Deep Dive)
================================================================

~52 advanced features from Baseball Savant that go far beyond barrel rate
and EV. Designed to capture the PHYSICS of how home runs actually happen.

Sources: Bat Tracking, Swing Path, Pitch Movement, Expected Stats,
         Batted Ball Profile, Catcher Framing, Count Leverage

Usage:
    from advanced_features import engineer_advanced_features
    df = engineer_advanced_features(statcast_df)
"""

import pandas as pd
import numpy as np

# Physics constants
HR_EV_SWEET = 103.0
HR_LA_LOW, HR_LA_HIGH = 20.0, 38.0
HR_LA_SWEET_LOW, HR_LA_SWEET_HIGH = 25.0, 32.0
BAT_SPEED_AVG = 70.0

# ============================================================================
# 1. BATTER POWER MECHANICS
# ============================================================================

def engineer_batter_power(df):
    """Batted ball profile features capturing HR ability."""
    f = pd.DataFrame(index=df.index)
    ev = df['launch_speed'].fillna(0)
    la = df['launch_angle'].fillna(0)

    # Continuous barrel quality (Gaussian centered on EV=105, LA=28)
    f['barrel_quality'] = (
        np.exp(-0.5 * ((ev - 105) / 6)**2) *
        np.exp(-0.5 * ((la - 28) / 5)**2)
    )

    # Hard-hit + elevated = the HR precursor
    f['hard_elevated'] = ((ev >= 95) & la.between(HR_LA_LOW, HR_LA_HIGH)).astype(float)

    # Sweet spot contact (EV 100+, LA 25-32)
    f['sweet_spot_contact'] = ((ev >= 100) & la.between(HR_LA_SWEET_LOW, HR_LA_SWEET_HIGH)).astype(float)

    # EV squared — rewards top-end power non-linearly
    f['ev_squared'] = ev**2 / 10000

    # Pull-air rate (pulled fly balls become HRs at 3-4x rate)
    if 'hc_x' in df.columns and 'hc_y' in df.columns:
        spray = np.degrees(np.arctan2(
            -(df['hc_x'].fillna(125.42) - 125.42),
            (204.27 - df['hc_y'].fillna(204.27))
        ))
        is_pull = np.where(df.get('stand', 'R') == 'L', spray > 15, spray < -15).astype(float)
        is_air = (la >= 20).astype(float)
        is_hard = (ev >= 95).astype(float)
        f['pull_air'] = is_pull * is_air
        f['pull_air_barrel'] = is_pull * is_air * is_hard  # Holy grail
        f['spray_angle'] = spray

    f['la_in_hr_zone'] = la.between(HR_LA_LOW, HR_LA_HIGH).astype(float)
    return f


# ============================================================================
# 2. BAT TRACKING (Statcast 2024+)
# ============================================================================

def engineer_bat_tracking(df):
    """Swing mechanics — the swing itself, not just the outcome."""
    f = pd.DataFrame(index=df.index)
    ev = df['launch_speed'].fillna(0)

    if 'bat_speed' in df.columns:
        bs = df['bat_speed'].fillna(BAT_SPEED_AVG)
        f['bat_speed'] = bs
        f['bat_speed_above_avg'] = bs - BAT_SPEED_AVG
        f['bat_speed_sq'] = (bs / BAT_SPEED_AVG)**2
        f['bat_speed_x_ev'] = (bs * ev) / 7500  # Impact energy proxy

    if 'swing_length' in df.columns and 'bat_speed' in df.columns:
        sl = df['swing_length'].fillna(7.0).clip(lower=5)
        f['swing_efficiency'] = df['bat_speed'].fillna(BAT_SPEED_AVG) / sl

    for col in ['squared_up_rate', 'blast']:
        if col in df.columns:
            f[col] = df[col].fillna(0)

    return f


# ============================================================================
# 3. PITCHER VULNERABILITY
# ============================================================================

def engineer_pitcher_vuln(df):
    """Why certain pitchers give up more HRs — the physics."""
    f = pd.DataFrame(index=df.index)
    velo = df['release_speed'].fillna(92)

    f['pitch_velo'] = velo
    f['velo_deficit'] = 96 - velo  # Positive = slow = hittable
    f['velo_deficit_sq'] = f['velo_deficit'].clip(lower=0)**2

    # VAA proxy — flatter approach = easier to lift for HRs
    if 'pfx_z' in df.columns and 'release_pos_z' in df.columns:
        ivb = df['pfx_z'].fillna(12)
        rel_z = df['release_pos_z'].fillna(5.8)
        f['vaa_proxy'] = -(rel_z * 2 - ivb / 6)
        f['ivb'] = ivb
        f['low_ivb'] = (ivb < 14).astype(float)  # Flat FB = HR-prone
        f['ride_deficit'] = (96 - velo) + (14 - ivb) / 3  # Worst combo

    if 'pfx_x' in df.columns:
        hb = df['pfx_x'].fillna(0)
        f['horizontal_break'] = np.abs(hb)
        if 'pfx_z' in df.columns:
            f['total_movement'] = np.sqrt(hb**2 + df['pfx_z'].fillna(12)**2)
            f['low_movement'] = (f['total_movement'] < 15).astype(float)

    if 'release_extension' in df.columns:
        ext = df['release_extension'].fillna(6.2)
        f['release_extension'] = ext
        f['short_extension'] = (ext < 6.0).astype(float)

    if 'release_spin_rate' in df.columns:
        f['spin_rate'] = df['release_spin_rate'].fillna(2200)
        f['spin_velo_ratio'] = f['spin_rate'] / velo.clip(lower=80)

    if 'pitch_type' in df.columns:
        f['is_fastball'] = df['pitch_type'].isin(['FF','SI','FC']).astype(float)
        f['is_breaking'] = df['pitch_type'].isin(['SL','CU','KC','ST','SV']).astype(float)

    # Zone location
    if 'plate_x' in df.columns and 'plate_z' in df.columns:
        px, pz = df['plate_x'].fillna(0), df['plate_z'].fillna(2.5)
        f['in_heart'] = ((np.abs(px) < 0.5) & pz.between(1.8, 3.2)).astype(float)
        f['belt_high'] = pz.between(2.0, 3.0).astype(float)
        if 'stand' in df.columns:
            f['pitch_inside'] = np.where(
                df['stand'] == 'R', px < -0.3, px > 0.3
            ).astype(float)

    return f


# ============================================================================
# 4. BATTER x PITCHER INTERACTIONS — Where no other model goes
# ============================================================================

def engineer_interactions(df, bat_f, pit_f):
    """
    Interaction features modeling the COLLISION between bat mechanics
    and pitch characteristics. HRs happen when swing plane matches pitch plane.
    """
    f = pd.DataFrame(index=df.index)

    # Swing plane x pitch plane alignment
    if 'vaa_proxy' in pit_f.columns:
        la = df['launch_angle'].fillna(0)
        f['plane_match'] = np.exp(-0.5 * ((la - 28 + pit_f['vaa_proxy']) / 8)**2)

    # Bat speed vs pitch velo differential
    if 'bat_speed' in bat_f.columns and 'pitch_velo' in pit_f.columns:
        f['speed_differential'] = (bat_f['bat_speed'] - 65)/10 - (pit_f['pitch_velo'] - 90)/8
        f['impact_energy'] = (bat_f['bat_speed']**2 + pit_f['pitch_velo']**2) / 20000

    # Barrel quality x velo interaction
    if 'barrel_quality' in bat_f.columns and 'pitch_velo' in pit_f.columns:
        f['damage_cocktail'] = bat_f['barrel_quality'] * (pit_f['pitch_velo'] / 95)**1.5

    # Pull tendency x inside pitch = HR spike
    if 'pull_air' in bat_f.columns and 'pitch_inside' in pit_f.columns:
        f['pull_inside_match'] = bat_f['pull_air'] * pit_f['pitch_inside']

    # Platoon interactions
    if 'stand' in df.columns and 'p_throws' in df.columns:
        platoon = (df['stand'] != df['p_throws']).astype(float)
        f['platoon'] = platoon
        if 'is_fastball' in pit_f.columns:
            f['platoon_x_fastball'] = platoon * pit_f['is_fastball']
        if 'velo_deficit' in pit_f.columns:
            f['platoon_x_velo_deficit'] = platoon * pit_f['velo_deficit'].clip(lower=0)

    # Mistake pitch to power hitter
    if 'in_heart' in pit_f.columns and 'ev_squared' in bat_f.columns:
        f['mistake_to_power'] = pit_f['in_heart'] * bat_f['ev_squared']

    return f


# ============================================================================
# 5. COUNT LEVERAGE — Hidden variable
# ============================================================================

def engineer_count_features(df):
    """Count leverage affects what pitches hitters see → affects HR rate."""
    f = pd.DataFrame(index=df.index)
    if 'balls' in df.columns and 'strikes' in df.columns:
        b, s = df['balls'].fillna(0), df['strikes'].fillna(0)
        f['hitter_count'] = ((b >= 2) & (s <= 1)).astype(float)
        f['three_one'] = ((b == 3) & (s == 1)).astype(float)
        f['count_leverage'] = (b - s).clip(-2, 3)
        f['two_strikes'] = (s == 2).astype(float)
    return f


# ============================================================================
# 6. COMPOSITES — Our secret sauce
# ============================================================================

def engineer_composites(bat_f, pit_f, int_f):
    """
    High-information-density composite features.
    Each answers a specific question about HR probability.
    """
    f = pd.DataFrame(index=bat_f.index)
    n = len(bat_f)

    bq = bat_f.get('barrel_quality', pd.Series(0.05, index=bat_f.index))
    ev_sq = bat_f.get('ev_squared', pd.Series(0.75, index=bat_f.index))
    pull = bat_f.get('pull_air', pd.Series(0.10, index=bat_f.index))

    # HR ceiling — max probability from batter mechanics alone
    f['hr_ceiling'] = (bq * 0.4 + ev_sq * 0.35 + pull * 0.25).clip(0, 1)

    # Pitcher exploitability
    vd = pit_f.get('velo_deficit', pd.Series(0, index=bat_f.index))
    low_ivb = pit_f.get('low_ivb', pd.Series(0, index=bat_f.index))
    low_move = pit_f.get('low_movement', pd.Series(0, index=bat_f.index))
    heart = pit_f.get('in_heart', pd.Series(0.08, index=bat_f.index))

    f['pitcher_exploitability'] = (
        vd.clip(0, 8)/8 * 0.30 + low_ivb * 0.25 +
        low_move * 0.20 + heart * 0.25
    ).clip(0, 1)

    # Master convergence score
    plane = int_f.get('plane_match', pd.Series(0.5, index=bat_f.index))
    platoon = int_f.get('platoon', pd.Series(0.5, index=bat_f.index))

    f['convergence_score'] = (
        f['hr_ceiling'] * 0.35 + f['pitcher_exploitability'] * 0.30 +
        plane * 0.20 + platoon * 0.15
    )

    # Calibrated HR probability from convergence
    base_rate = 0.031
    f['adj_hr_prob'] = (base_rate * np.exp(4.0 * (f['convergence_score'] - 0.5))).clip(0.005, 0.15)

    return f


# ============================================================================
# MASTER FUNCTION
# ============================================================================

def engineer_advanced_features(df, include_bat_tracking=True):
    """Run all feature engineering. Returns ~52 features per PA."""
    print(f"Engineering advanced features for {len(df):,} PAs...")

    bat_f = engineer_batter_power(df)
    print(f"  Batter power: {len(bat_f.columns)} features")

    track_f = engineer_bat_tracking(df) if include_bat_tracking else pd.DataFrame(index=df.index)
    print(f"  Bat tracking: {len(track_f.columns)} features")

    pit_f = engineer_pitcher_vuln(df)
    print(f"  Pitcher vulnerability: {len(pit_f.columns)} features")

    all_bat = pd.concat([bat_f, track_f], axis=1)
    int_f = engineer_interactions(df, all_bat, pit_f)
    print(f"  Interactions: {len(int_f.columns)} features")

    count_f = engineer_count_features(df)
    print(f"  Count leverage: {len(count_f.columns)} features")

    comp_f = engineer_composites(all_bat, pit_f, int_f)
    print(f"  Composites: {len(comp_f.columns)} features")

    result = pd.concat([bat_f, track_f, pit_f, int_f, count_f, comp_f], axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    print(f"  TOTAL: {len(result.columns)} advanced features")
    return result


if __name__ == "__main__":
    print("Advanced Feature Engineering Module")
    print("="*50)
    print("\nExpected top features by importance:")
    print("  1. barrel_quality       — continuous barrel score")
    print("  2. convergence_score    — master composite")
    print("  3. ev_squared           — non-linear EV reward")
    print("  4. pull_air_barrel      — pulled + elevated + hard")
    print("  5. velo_deficit         — slow pitchers = more HRs")
    print("  6. plane_match          — swing plane vs pitch plane")
    print("  7. damage_cocktail      — barrel quality x pitch velo")
    print("  8. ride_deficit         — low velo + low IVB = BP FB")
    print("  9. platoon_x_fastball   — platoon edge on FBs")
    print(" 10. mistake_to_power     — heart-zone to power hitter")
