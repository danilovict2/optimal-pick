# Optimal Pick

HackPrinceton project: an ML + optimization pipeline that projects NFL fantasy
points, optimizes lineups with an ILP, drafts with a Monte Carlo bot in 12-team
snake drafts, and is **backtested on the real 2025 season** (trained only on
data through 2024). Includes a Streamlit mock-draft app with pick grades.

## How to run

```bash
uv sync
uv run python data_nfl.py
uv run python projections.py
uv run python optimizer.py
uv run python draft_sim.py
uv run python backtest.py
uv run python grades.py
uv run streamlit run app.py
```

## Method

1. **Projections** — weekly stats from `nflreadpy` aggregated to player-season
   rows for all official-app positions (QB/RB/WR/TE/K/DST); a
   `HistGradientBoostingRegressor` predicts season-total PPR points
   from the previous two seasons' usage/production (strictly no leakage:
   features for season *y* use only seasons < *y*). Kicker points are
   computed from FG distances (0–39: 3, 40–49: 4, 50+: 5, PAT 1, miss −1);
   D/ST points use standard scoring (sacks, turnovers, TDs, safeties,
   blocks, points-allowed tiers from final scores).
2. **Lineup optimizer** — assignment-form ILP (PuLP/CBC) over the official
   lineup (QB, 2 RB, 2 WR, TE, FLEX, K, D/ST); a fast greedy evaluator
   (verified identical to the ILP on 200 random rosters) is used inside
   simulations.
3. **Draft bot** — at each of our picks, the top-8 candidates by value-over-
   replacement are each evaluated by Monte Carlo rollouts of the rest of the
   draft (opponents = VOR bots with noise and roster caps); we pick the
   candidate with the highest mean final roster value. < 3 s per pick.
4. **Backtest** — 1000 seeded drafts per strategy from random slots; every
   drafted team is scored **week by week**: for each 2025 regular-season week,
   the roster's best legal lineup from that week's **actual PPR points**
   (byes, mid-season injuries, and weekly variance all count), summed over
   the season. Compared against best-projected-available and the VOR bot.
   Backtest uses R=16 rollouts per candidate (app default 64) to keep runtime
   reasonable. Draft bots also mirror real behavior: K/D/ST wait until the
   late rounds and every roster is forced to fill a legal starting lineup.
5. **Grades** — pick grade blends projection percentile within position with
   value vs the average player taken at that pick in simulated drafts; draft
   grade is your roster-value percentile vs simulated drafts from your slot.
6. **Leagues** — a finished draft is saved as a league (JSON under
   `leagues/`). The My Team tab loads it, re-optimizes your lineup for the
   current week from each player's recent form, and shows projected scores
   for every team in the league. Weekly lineups respect bye weeks and the
   latest injury report (Out/Doubtful sit, Questionable is flagged); players
   with no recent games are benched and a waiver warning appears if a slot
   has no healthy option.
7. **Always current** — the app auto-detects the latest season in the data:
   it trains on every completed season and projects the next draftable one
   (today: trained through 2025, projecting 2026; once 2026 ends it will
   project 2027 with no code change). The in-progress season's stats refresh
   daily for the Start/Sit tab. The backtest stays pinned to 2025 by design —
   it's a historical validation.

The engine is league-agnostic: optimizer and draft sim only see players
(id, name, position, projection) + a config (`config.py`). Adding NBA = a new
data/projection module + a config entry.
