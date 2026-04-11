# MLB Home Run Prop Prediction Model

A data-driven HR prop betting model built on Statcast data, designed to find +EV home run bets by comparing calibrated probability estimates against sportsbook odds.

## Philosophy

This model is built on principles from sharp bettors and proven model architectures:

- **Top-down edge detection** (GiuseppePaps): Find discrepancies vs the market, don't try to outsmart it on "why" a bet hits. The model's job is to produce well-calibrated probabilities, then compare them against what books are offering.

- **Batter + Pitcher independently > matchup specifics** (GiuseppePaps research): Modeling batter-vs-pitch-arsenal is mostly noise. Knowing the batter's power profile and the pitcher's HR vulnerability separately predicts better than granular matchup history.

- **Hitter-level park factors** (BallparkPal methodology): Park effects should be personalized — a pull-heavy lefty experiences Yankee Stadium differently than a spray hitter. BallparkPal's dual-model approach (Contact Only vs Contact+Park+Weather) is the gold standard.

- **Isotonic calibration** (Monte Carlo backtest): Platt scaling is too rigid for the extreme probability tails where HR props live. Isotonic regression dramatically improves calibration in the 1-8% range.

- **Confidence tiering** (Monte Carlo backtest): Top 20% of picks by edge size returned +6.1% ROI vs +2.1% for all picks. Selectivity matters enormously.

## Project Structure

```
mlb_hr_model/
├── phase1_data_pipeline.py    # Data acquisition + feature engineering
├── phase2_model_training.py   # Model training + calibration + evaluation
├── run_model.py               # Quickstart runner (ties everything together)
├── data/                      # Cached Statcast data (auto-created)
│   └── ballparkpal/           # BallparkPal subscription exports (manual)
├── models/                    # Trained model artifacts (auto-created)
└── output/                    # Predictions and reports (auto-created)
```

## Setup

```bash
# 1. Install dependencies
pip install pybaseball pandas numpy scikit-learn lightgbm

# 2. Quick test (downloads ~1 month of data, runs in ~5 min)
python run_model.py --test

# 3. Full training run (downloads 2 seasons, takes 30-60 min)
python run_model.py

# 4. Subsequent runs with cached data
python run_model.py --from-cache
```

## Feature Set (V1: ~25 features)

### Batter Power Profile
| Feature | Description | Source |
|---------|-------------|--------|
| Barrel rate | % of batted balls that are "barreled" (98+ EV, optimal LA) | Statcast |
| Avg exit velocity | Mean EV on contact | Statcast |
| Avg launch angle | Mean LA — optimal HR range is 25-35° | Statcast |
| HR/PA rate | Season HR rate per plate appearance | Statcast |
| Fly ball rate | % of batted balls classified as fly balls | Statcast |
| Hard hit rate | % of batted balls 95+ mph | Statcast |
| Pull rate | % of batted balls pulled (where HRs concentrate) | Statcast |
| ISO | Isolated power (SLG - AVG) | Computed |
| HR form delta | Recent HR rate minus season rate (streak detection) | Computed |
| Barrel form delta | Recent barrel rate minus season rate | Computed |

### Pitcher HR Vulnerability
| Feature | Description | Source |
|---------|-------------|--------|
| HR/PA rate allowed | How often this pitcher gives up HRs | Statcast |
| FB rate allowed | Fly ball tendency = more HR opportunities | Statcast |
| Hard hit rate allowed | Quality of contact allowed | Statcast |
| Barrel rate allowed | Barrels per PA allowed | Statcast |
| Avg EV allowed | Mean exit velocity allowed | Statcast |
| FB velocity | Fastball speed (stuff quality proxy) | Statcast |
| VAA proxy | Vertical approach angle estimate | Computed |
| HR form delta | Recent HR rate vs season (trending worse/better) | Computed |

### Environment
| Feature | Description | Source |
|---------|-------------|--------|
| Platoon advantage | Batter handedness vs pitcher handedness | Statcast |
| Venue | Park ID (tree model learns park effects) | Statcast |
| HR park factor | Hitter-specific park factor for HRs | BallparkPal |
| Temperature | Game-day temperature | Weather/BPP |
| Wind speed | Game-day wind | Weather/BPP |

## Roadmap

- [x] Phase 1: Data pipeline + feature engineering
- [x] Phase 2: Model training + isotonic calibration
- [ ] Phase 3: Historical odds backtest + ROI simulation
- [ ] Phase 4: Live daily prediction pipeline
- [ ] BallparkPal integration (hitter-level park factors)
- [ ] Catcher framing feature
- [ ] Umpire tendency feature
- [ ] Expected PA estimation (lineup position + game total)
- [ ] Automated odds API integration
