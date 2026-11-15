# Optimal Pick

A fantasy draft war room for **NFL, NBA, and the Premier League**: ML-projected fantasy points, ILP-optimized lineups, a Monte Carlo draft bot, pick and draft grades, weekly start/sit advice — all backtested against real completed seasons.

## 🚀 Quick Start

Ensure you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed (or [Docker](https://www.docker.com/get-started), see below).

### Clone the project

```bash
git clone https://github.com/danilovict2/optimal-pick.git
cd optimal-pick
```

### Run the app

```bash
uv sync
uv run streamlit run app.py
```

### Access it

Open [http://localhost:8501](http://localhost:8501). Pick a league in the sidebar, pick your draft slot, and hit **Start new draft**.

> The first load of each league downloads its data and trains the model (a minute or so). After that everything is cached — past seasons forever, the current season refreshed daily.

### Run with Docker (alternative)

```bash
docker compose up -d --build
```

Then open [http://localhost](http://localhost).

## ✨ Features

- 🏈🏀⚽ **Three leagues, one engine** — NFL (PPR, full K/D/ST lineup), NBA (points league, G/F/C), Premier League (official FPL Draft rules, exact 2/5/5/3 squads).
- 🤖 **Monte Carlo draft bot** — every recommendation comes from simulating the rest of your draft dozens of times, in under ~2 s per pick.
- 🎙️ **Live draft helper** — enter the real picks as they happen in your actual draft and get told who to take.
- 📝 **Pick & draft grades** — every pick graded against thousands of simulated drafts; a letter grade for your whole draft at the end.
- 📋 **My Team** — finished drafts are saved as leagues; get your optimal lineup each week from recent form, injuries, byes, and availability flags.
- 📈 **Honest backtests** — drafts replayed against what players *actually did*, scored week by week with real lineup rules.

## 📖 Usage

Everything lives in the app, but each stage is also a standalone CLI. League is `nfl` | `nba` | `fpl` (default `nfl`):

| Command | What it does |
| --- | --- |
| `uv run python data_nfl.py` | Download + cache a league's data (also `data_nba.py`, `data_fpl.py`) |
| `uv run python projections.py nba` | Train the model, print validation MAE and the projected top players |
| `uv run python draft_sim.py fpl` | Simulate a full draft and time a Monte Carlo pick |
| `uv run python backtest.py nfl 1000` | Run the full backtest (N drafts per strategy, parallel) |
| `uv run python optimizer.py` | Verify the fast greedy evaluator matches the ILP in every league |
| `uv run python grades.py` | Grade a simulated draft end to end |

### League formats

| | NFL | NBA | Premier League |
|---|---|---|---|
| Format | PPR, 12-team | Points (Yahoo-style), 10-team | Official FPL Draft, 8-team |
| Lineup | QB, 2 RB, 2 WR, TE, FLEX, K, D/ST | 3 G, 3 F, C, 3 UTIL | XI: GKP + 3–5 DEF + 2–5 MID + 1–3 FWD |
| Roster | 15 | 13 | 15 (exactly 2 GKP / 5 DEF / 5 MID / 3 FWD) |
| Data | [nflverse](https://github.com/nflverse) via `nflreadpy` | stats.nba.com via `nba_api` | official FPL API + [vaastav archive](https://github.com/vaastav/Fantasy-Premier-League) |

League size is adjustable in the app; scoring is handled per sport (NFL kicker points from FG distance, D/ST from standard defensive scoring; NBA = PTS + 1.2·REB + 1.5·AST + 3·STL + 3·BLK − TO; FPL points are the official ones).

## 🧠 How it works

1. **Projections** — per-game stats roll up to player-season rows; a `HistGradientBoostingRegressor` per league predicts season-total fantasy points from the previous two seasons (strictly no leakage: features for season *y* use only seasons < *y*). Newcomers get a low position-specific default.
2. **Lineup optimizer** — an assignment-form ILP (PuLP/CBC) over each league's slots, flex spots, and formation caps; a fast greedy evaluator, verified identical to the ILP on 200 random rosters per league, runs inside simulations.
3. **Draft bot** — at each of your picks, the top-8 candidates by value-over-replacement are evaluated by Monte Carlo rollouts of the rest of the draft against realistic opponents (VOR bots with noise, league-specific roster rules, late-round K/D/ST habits, forced legal lineups).
4. **Backtest** — 1000 seeded drafts per strategy from random slots, every roster scored week by week with its best legal lineup from **actual** weekly points, so byes, injuries, and variance all count.

## 📊 Backtest results

2025 (NFL) and 2025–26 (NBA, PL) seasons, 1000 drafts per strategy, trained only on earlier data:

| League | Monte Carlo (ours) | VOR bot | Best available |
|---|---|---|---|
| NFL (of 12) | **6.0 · top-3 27%** | 6.5 · 24% | 8.0 · 7% |
| NBA (of 10) | 6.4 · 18% | **5.4 · 32%** | 6.6 · 16% |
| Premier League (of 8) | **4.3 · 1st 16%** | 4.4 · 13% | 4.4 · 11% |

The pattern is itself a finding: Monte Carlo planning pays off most where positions are scarce and heterogeneous (NFL), while in structurally constrained drafts (FPL's fixed quotas) or homogeneous ones (NBA) simple value-over-replacement is already near-optimal.

## 🏗️ Architecture

The engine is league-agnostic: the optimizer, draft sim, and grader only ever see players (id, name, position, projection) plus a config. Adding a sport = one data module implementing a small contract (positions, lag stats, player-seasons, weekly points, current form) + one entry in `config.py`.

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
