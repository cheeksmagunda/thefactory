# The Factory

MLB home run prop prediction engine. Runs once daily at **9:00 AM Central**,
scores every eligible batter on the slate using the composite **Nuke E Score**,
selects the top 3 picks (max 1 per team), caches them in Redis, and serves them
from a single API endpoint.

**If any data source fails, the entire run aborts — no stale data, no fallbacks, no mocks.**

## Layout

```
src/
  main.py                     FastAPI app + scheduler + /picks endpoint
  config.py                   Pydantic settings
  cache/redis_client.py       Pick + status + daily-lock caching
  models/schemas.py           Pydantic models (Pick, BatterScore, PipelineStatus)
  pipeline/
    __init__.py               PipelineError / DataSourceError
    orchestrator.py           Daily run: fetch → score → select → cache
    data_sources/
      rotowire.py             Confirmed lineups + SPs + weather + O/U (STUB)
      ballparkpal.py          BPP XLSX export (38 columns)
      fangraphs.py            Bat tracking via pybaseball (weekly cache)
      statcast.py             Pitcher profiles via pybaseball (weekly cache)
      odds.py                 HR prop odds (FUTURE)
    scoring/
      bat_tracking.py         HR Physical Ceiling
      bpp_features.py         BPP per-component scores + tiers
      pitcher_vuln.py         Statcast-based pitcher vulnerability
      nuke_e.py               Weighted composite score
    selection/picker.py       Top-3 with max-1-per-team diversification
tests/                        Unit tests
```

## API

`GET /picks` — today's picks

```json
{
  "date": "2026-04-21",
  "status": "cached",
  "picks": [
    {
      "batter_name": "Aaron Judge",
      "team": "NYY",
      "opponent_team": "BOS",
      "opponent_pitcher": "...",
      "nuke_e_score": 63.12,
      "bpp_hr_prob": 4.1,
      "validation": 4,
      "rationale": "BPP HR Prob 4.1%; vs Grade +6; elite bat tracking (22)"
    }
  ]
}
```

Other endpoints:

* `GET /health` — health + redis ping
* `POST /picks/run` — manually trigger the pipeline (ops/retry)

Statuses returned by `/picks`:

| status    | meaning                                          |
| --------- | ------------------------------------------------ |
| `cached`  | picks available                                  |
| `pending` | scheduler hasn't fired yet today                 |
| `failed`  | scheduled run failed — see `error`               |

## Local dev

```bash
pip install -r requirements.txt
cp .env.example .env           # fill in REDIS_URL

# Trigger the pipeline (will fail until data sources are wired up)
uvicorn src.main:app --reload
```

`POST /picks/run` executes the pipeline synchronously. Until each data source
is implemented, that call will return a 503 with a descriptive `DataSourceError`.

## Railway deploy

1. Push to the configured branch; Railway auto-deploys via `railway.toml` / Procfile.
2. Attach the **Redis plugin** in the dashboard — `REDIS_URL` is injected.
3. Set any additional env vars from `.env.example` as needed.
4. The scheduler fires at `SCHEDULE_HOUR_UTC:SCHEDULE_MINUTE_UTC` (default
   `14:00 UTC` == 9:00 AM CT).

## Data source status

| Source         | Status      | Notes                                                         |
| -------------- | ----------- | ------------------------------------------------------------- |
| RotoWire       | stub        | HTML scrape — extract teams, SPs, batting orders, weather, O/U |
| BallparkPal    | manual path | Drop `Matchups_YYYY-MM-DD.xlsx` in `data/ballparkpal/`         |
| FanGraphs      | working     | `pybaseball.batting_stats()`, weekly cache                     |
| Statcast       | working     | `pybaseball.pitching_stats()`, weekly cache                    |
| Odds           | stub        | Returns empty dict; edge calc deferred                         |

A `DataSourceError` aborts the whole run and `/picks` reports
`{status: "failed", error: "..."}`. That's the intended failure mode.

## Nuke E Score (v1 weights)

```
nuke_e = 0.25 * BPP simulation score
       + 0.15 * BPP matchup quality (no park)
       + 0.10 * environment boost
       + 0.10 * vs Grade
       + 0.15 * HR physical ceiling (bat tracking)
       + 0.05 * contact quality
       + 0.10 * pitcher vulnerability (Statcast)
       + 0.10 * validation bonus (independent signals agreeing)
```

Weights live in `src/pipeline/scoring/nuke_e.py::WEIGHTS` and will be tuned
against graded pick history.

## Selection rules

* Exactly 3 picks
* Max 1 batter per team
* Starters only
* Sort by Nuke E score, descending — pick until 3 distinct teams filled
* If fewer than 3 eligible batters on the slate → `PipelineError`

## Tests

```bash
pip install pytest
pytest tests/ -v
```
