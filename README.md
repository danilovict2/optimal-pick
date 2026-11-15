# Optimal Pick

An ML + optimization pipeline for fantasy drafts across **NFL, NBA, and the
Premier League (FPL Draft)**: it projects fantasy points, optimizes lineups
with an ILP, drafts with a Monte Carlo bot in snake drafts, and is
**backtested on real completed seasons** (trained only on earlier data).
Includes a Streamlit mock-draft app with pick grades and weekly start/sit.

## How to run

```bash
uv sync
uv run streamlit run app.py
```

Module entry points (each takes a league: `nfl` | `nba` | `fpl`):

```bash
uv run python data_nfl.py          # or data_nba.py / data_fpl.py
uv run python projections.py nfl
uv run python optimizer.py         # verifies greedy == ILP for every league
uv run python draft_sim.py nfl
uv run python backtest.py nfl 1000
uv run python grades.py
```

## Leagues

| | NFL | NBA | Premier League |
|---|---|---|---|
| Format | PPR, 12-team | Points (Yahoo-style), 10-team | Official FPL Draft, 8-team |
| Lineup | QB, 2 RB, 2 WR, TE, FLEX, K, D/ST | 3 G, 3 F, C, 3 UTIL | XI: GKP + 3-5 DEF + 2-5 MID + 1-3 FWD |
| Roster | 15 | 13 | 15 (exactly 2/5/5/3 — enforced) |
| Data | `nflreadpy` (nflverse) | `nba_api` (stats.nba.com) | official FPL API + vaastav archive |
| Backtest | 2025 | 2025-26 | 2025-26 |

Sport-specific scoring handled in the data modules: NFL kicker points from FG
distances and D/ST from standard defensive scoring; NBA fantasy points =
PTS + 1.2·REB + 1.5·AST + 3·STL + 3·BLK − TO; FPL points are the official ones.

## Method

1. **Projections** — per-game/per-week stats aggregated to player-season
   rows; one `HistGradientBoostingRegressor` per league predicts
   season-total fantasy points from the previous two seasons' usage and
   production (strictly no leakage: features for season *y* use only
   seasons < *y*). Newcomers to a league (NFL/NBA rookies, new PL signings)
   get a low position-specific default (30th-percentile projection).
2. **Lineup optimizer** — assignment-form ILP (PuLP/CBC) over the league's
   slots, flex slots, and formation caps (FPL's 5-DEF/5-MID/3-FWD limits);
   a fast greedy evaluator (verified identical to the ILP on 200 random
   rosters per league) is used inside simulations.
3. **Draft bot** — at each of our picks, the top-8 candidates by value-over-
   replacement are each evaluated by Monte Carlo rollouts of the rest of the
   draft (opponents = VOR bots with noise and league-specific roster rules);
   we pick the candidate with the highest mean final roster value. < 3 s per
   pick. Bots mirror real behavior: NFL K/D/ST wait until the late rounds,
   FPL squads fill their exact quotas, and every roster is forced to
   complete a legal starting lineup.
4. **Backtest** — seeded drafts per strategy from random slots; every
   drafted team is scored **week by week**: for each scoring period of the
   backtest season, the roster's best legal lineup from that period's
   **actual fantasy points** (byes, absences, and weekly variance all
   count), summed over the season. Compared against best-projected-available
   and the VOR bot. Uses R=16 rollouts per candidate (app default 64) to
   keep runtime reasonable.
5. **Grades** — pick grade blends projection percentile within position with
   value vs the average player taken at that pick in simulated drafts; draft
   grade is your roster-value percentile vs simulated drafts from your slot.
6. **Leagues** — a finished draft is saved per sport (JSON under
   `leagues/`). The My Team tab loads it, re-optimizes your lineup for the
   current period from each player's recent form, and shows projected scores
   for every team. NFL lineups respect bye weeks and the injury report; NBA
   benches players without recent games; FPL uses the official form and
   availability flags. A waiver warning appears if a slot has no available
   option.
7. **Always current** — the app auto-detects the latest season per league:
   it trains on every completed season and projects the next draftable one,
   with no code change at season rollover. In-progress seasons refresh
   daily. Backtests stay pinned to completed seasons by design — they're
   historical validations.

The engine is league-agnostic: optimizer and draft sim only see players
(id, name, position, projection) plus a config (`config.py`). Adding a new
sport = a data module exposing the small data contract (positions, lag
stats, player-seasons, weekly points, current form) + a config entry.
