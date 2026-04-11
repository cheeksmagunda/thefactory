# MLB HR Model — Data Pipeline & Source Map
## Single source of truth for all data inputs

Last updated: April 11, 2026

---

## DAILY WORKFLOW (pre-game, ~3 hours before first pitch)

### Step 1: Get today's slate
- **Source:** [RotoWire Daily Lineups](https://www.rotowire.com/baseball/daily-lineups.php)
- **What we pull:** Confirmed lineups, starting pitchers (name + handedness + ERA), weather, wind speed/direction, O/U lines, sportsbook odds
- **Frequency:** Check once in morning for probables, re-check 2-3 hours before first pitch for confirmed lineups
- **CRITICAL:** This is our ONLY source for daily lineups and starting pitchers. Do not guess or use projected lineups from other sources.

### Step 2: Get HR prop odds
- **Source:** [FanDuel HR Props](https://www.fanduel.com/research/mlb-home-run-prop-odds) (primary)
- **Backup:** DraftKings, BetMGM for line shopping
- **What we pull:** American odds for each hitter to hit 1+ HR
- **Frequency:** Updated throughout the day. Check after lineups are confirmed for sharpest prices.

### Step 3: Identify vulnerable pitchers
- **Source:** [Baseball Savant](https://baseballsavant.mlb.com/) for Statcast pitcher profiles
- **Source:** [FanGraphs Pitching Stats](https://www.fangraphs.com/leaders/major-league?pos=all&stats=pit&type=8) for ERA, xERA, FIP, HR/9
- **What we look for:** High ERA + high HR/9 + high hard-hit rate allowed + high FB rate allowed + low fastball velo
- **Red flags (target these pitchers):** ERA > 4.50, HR/9 > 1.3, Hard-hit% allowed > 40%, career ERA significantly higher than current small-sample ERA

### Step 4: Score hitters against vulnerable pitchers
- **Source:** [Baseball Savant Batter Profiles](https://baseballsavant.mlb.com/) for barrel rate, EV, ISO, launch angle
- **Source:** [FanGraphs Batter Splits](https://www.fangraphs.com/leaders/splits-leaderboards) for L/R splits, home/away splits
- **What we look for:** High barrel rate + high EV + high ISO + recent form (hot streak) + platoon advantage

### Step 5: Environment check
- **Source:** [RotoWire Daily Lineups](https://www.rotowire.com/baseball/daily-lineups.php) — weather, wind, dome status
- **Source:** [BallparkPal Park Factors](https://www.ballparkpal.com/Park-Factors-Preview.php) — hitter-level park factors (subscription)
- **What we check:** Temperature (>65F = good), wind direction (out = good, in = bad), altitude, park HR factor

### Step 6: Classify picks and generate pairings
- **Framework:** Triple Convergence (power + mispricing + environment)
- **Odds sweet spot:** +350 to +700 range (where books are least efficient)
- **Pairing logic:** Same-team stacks against vulnerable pitchers, cross-game independent pairings

---

## DATA SOURCES — FULL REFERENCE

### Tier 1: Daily / Real-Time (check every game day)

| Source | URL | What We Get | Free? |
|--------|-----|-------------|-------|
| RotoWire Lineups | https://www.rotowire.com/baseball/daily-lineups.php | Confirmed lineups, SPs, weather, odds, O/U | Partial (basic free) |
| FanDuel Research | https://www.fanduel.com/research/ | HR prop odds, player stats | Free |
| MLB.com Probable Pitchers | https://www.mlb.com/probable-pitchers | Official SP announcements | Free |
| Baseball Savant | https://baseballsavant.mlb.com/ | Statcast data, barrel rates, EV, sprint speed | Free |
| BallparkPal Cheat Sheets | https://www.ballparkpal.com/Cheat-Sheets.php | Daily park factors, HR zone, pitcher intel | Subscription |

### Tier 2: Statistical Profiles (check weekly / as needed)

| Source | URL | What We Get | Free? |
|--------|-----|-------------|-------|
| FanGraphs Leaders | https://www.fangraphs.com/leaders/major-league | Season stats, splits, advanced metrics | Partial |
| Baseball Savant Leaderboards | https://baseballsavant.mlb.com/leaderboard | Barrel%, EV, xSLG, Hard-hit% | Free |
| Statcast Search | https://baseballsavant.mlb.com/statcast_search | Custom queries on pitch/PA data | Free |
| BallparkPal Methods | https://www.ballparkpal.com/Methods.php | Park factor methodology | Free |

### Tier 3: Model Training (run periodically to retrain)

| Source | URL | What We Get | Free? |
|--------|-----|-------------|-------|
| pybaseball (Statcast) | `pip install pybaseball` | Historical PA-level data for model training | Free |
| pybaseball (FanGraphs) | Same package | Season-level batting/pitching stats | Free |
| BallparkPal Export | Subscription portal | Hitter-level daily park factors (CSV) | Subscription |

### Tier 4: Sharp Model Tracking (ongoing)

| Source | URL | What We Track | Free? |
|--------|-----|---------------|-------|
| GiuseppePaps | https://x.com/GiuseppePaps | HR picks, methodology, win slips | Free |
| HomeRunPredict | https://x.com/HomeRunPredict | Daily HR projections | Free |
| Covers HR Props | https://www.covers.com/mlb/home-run-props-parlay-and-odds | Expert HR analysis | Free |
| BallparkPal HR Zone | https://www.ballparkpal.com/Home-Run-Zone.php | Most likely HR hitters | Subscription |

---

## HOW DATA INTERACTS IN THE MODEL

```
                    DAILY INPUTS
                    ============
    RotoWire                FanDuel              Baseball Savant
    (lineups,               (HR prop             (batter/pitcher
     SPs, weather)           odds)                profiles)
         |                     |                      |
         v                     v                      v
    ┌─────────────────────────────────────────────────────┐
    │           FEATURE ENGINEERING LAYER                  │
    │                                                      │
    │  Batter: barrel%, EV, ISO, LA, FB%, pull%, form     │
    │  Pitcher: ERA, HR/9, hard-hit%, FB velo, FB%, VAA   │
    │  Environment: temp, wind, park factor, O/U, platoon │
    └──────────────────────┬──────────────────────────────┘
                           │
                           v
    ┌─────────────────────────────────────────────────────┐
    │              SCORING LAYER                           │
    │                                                      │
    │  Power Score (0-100)     ← Can he actually go yard? │
    │  Pitcher Vuln (0-100)    ← How hittable is the SP?  │
    │  Env Score (0-100)       ← Weather/park helping?    │
    │  → Model HR Probability  ← Composite game-level %   │
    └──────────────────────┬──────────────────────────────┘
                           │
                           v
    ┌─────────────────────────────────────────────────────┐
    │              EDGE DETECTION LAYER                    │
    │                                                      │
    │  Model Prob vs Market Implied Prob (from FanDuel)   │
    │  Edge % = Model - Market (after vig removal)        │
    │  Filter: +3% edge minimum for consideration         │
    │  Odds sweet spot: +350 to +700                      │
    └──────────────────────┬──────────────────────────────┘
                           │
                           v
    ┌─────────────────────────────────────────────────────┐
    │              CLASSIFICATION LAYER                    │
    │                                                      │
    │  TRIPLE CONVERGENCE: Power ≥55, Vuln ≥55,          │
    │                       Env ≥55, Edge ≥3%             │
    │  VALUE + POWER: Power ≥45, Edge ≥3%                 │
    │  POWER PLAY: Power ≥60, Edge ≥1%                    │
    │  SKIP: Negative edge or low power                   │
    └──────────────────────┬──────────────────────────────┘
                           │
                           v
    ┌─────────────────────────────────────────────────────┐
    │              OUTPUT                                   │
    │                                                      │
    │  Top 3 official picks (ranked)                      │
    │  Best pairings (SGP + cross-game)                   │
    │  Avoid list (overpriced / tough matchups)           │
    └─────────────────────────────────────────────────────┘
```

---

## GIUSEPPE-DERIVED RULES (from Working Backwards analysis)

1. **Pitcher vulnerability is the primary target selector** — find the worst SP, then find the best-priced hitter facing him
2. **Deepest reasonable odds on a vulnerable pitcher** — if Schwarber is +270 and Marte is +520 facing the same pitcher, take Marte
3. **Game total (O/U) as a confirming signal** — O/U ≥ 8.5 = green light for HR hunting
4. **Series persistence** — when a series environment is favorable, press it across multiple days
5. **Small-sample ERA traps** — ignore 2-3 start ERA; use career ERA and underlying metrics (xERA, FIP)
6. **Hot streaks are underpriced at +400+** — books are slow to adjust for non-stars on heaters
7. **Profit boosts + SGP mechanics** — use near-lock filler legs (pitcher 1+ K) to hit SGP minimums and unlock boosts
8. **Round-robin 2x3 structure** — play all 3 straight + every 2-leg combo for bankroll-efficient coverage
9. **Never pay chalk for HRs** — the edge lives at +350 to +700, not +180 to +300

---

## FILE LOCATIONS (local)

```
~/mlb_hr_model/mlb_hr_model/
├── phase1_data_pipeline.py     # Statcast data pull + feature engineering
├── phase2_model_training.py    # LightGBM training + isotonic calibration
├── daily_picks.py              # Daily prediction engine
├── working_backwards.py        # Sharp model tracker
├── run_model.py                # Master training runner
├── data/                       # Cached Statcast data + BallparkPal CSVs
│   └── working_backwards_log.json
├── models/                     # Trained model artifacts
└── output/                     # Daily pick CSVs + performance tracking
```

---

## KNOWN GAPS (to build next)

- [ ] Automated RotoWire scraper for daily lineups + SPs
- [ ] Odds API integration for real-time HR prop prices
- [ ] BallparkPal daily park factor integration
- [ ] Catcher framing feature
- [ ] Umpire tendency feature
- [ ] Game total (O/U) as a model feature
- [ ] Career ERA vs current ERA weighting for early season
- [ ] Automated performance tracking + model recalibration
- [ ] SGP/boost optimizer for FanDuel parlay construction
