"""Backtest on a real season: train strictly on earlier seasons, draft as if
preseason, then score every team week by week: for each scoring period of
the backtest season, the roster's best legal lineup from that period's
actual fantasy points (so byes, absences, and weekly variance all count),
summed over the season.

Runs N drafts per strategy (our seat: Monte Carlo / best-available / VOR
bot; the other seats: VOR bots). MC uses R=16 rollouts here (vs 64 in the
app) to keep runtime down; drafts run in parallel across CPU cores.

Usage: python backtest.py [nfl|nba|fpl] [n_drafts]
"""

import os
import sys
import time
from multiprocessing import Pool as MPPool

import numpy as np
import pandas as pd

from config import LEAGUES
from draft_sim import make_pool, run_draft
from optimizer import greedy_lineup_value
from projections import league_engine, league_module

MC_R = 16
MC_K = 8

_G = {}


def _init(pool, actual):
    _G["pool"], _G["actual"] = pool, actual


def _actual_value(pool, actual, roster):
    """Sum over weeks of the roster's best lineup from that week's actual
    points. actual is a (players, weeks) matrix aligned to the pool."""
    r = np.asarray(roster, dtype=int)
    codes = pool["pos_codes"][r]
    return sum(
        greedy_lineup_value(actual[r, w], codes, pool["league"],
                            pool["pos_order"])
        for w in range(actual.shape[1]))


def _one_draft(args):
    strategy, seed = args
    pool, actual = _G["pool"], _G["actual"]
    slot = int(np.random.default_rng(seed).integers(0, pool["league"]["teams"]))
    rosters, _ = run_draft(pool, seed=seed, our_team=slot,
                           our_strategy=strategy, K=MC_K, R=MC_R)
    vals = [_actual_value(pool, actual, ro) for ro in rosters]
    rank = 1 + sum(v > vals[slot] for v in vals)
    return {"strategy": strategy, "seed": seed, "slot": slot + 1,
            "actual_points": vals[slot], "rank": rank}


def build_pool_and_actuals(league_key: str):
    cfg = LEAGUES[league_key]
    season = cfg["backtest_season"]
    _, pool_df, _ = league_engine(league_key, backtest_season=season)
    pool = make_pool(pool_df, cfg)

    # (players, weeks) matrix of actual weekly points, pool-aligned;
    # byes and missed games are simply 0 that week
    wk = league_module(league_key).load_weekly_points(season)
    mat = wk.pivot_table(index="player_id", columns="week",
                         values="fantasy_points", aggfunc="sum")
    actual = mat.reindex(pool["df"]["player_id"]).fillna(0.0).to_numpy(float)
    return pool, actual


def main():
    league_key = sys.argv[1] if len(sys.argv) > 1 else "nfl"
    n_drafts = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    cfg = LEAGUES[league_key]
    label = league_module(league_key).season_label(cfg["backtest_season"])

    pool, actual = build_pool_and_actuals(league_key)
    strategies = ["mc", "best_available", "vor"]
    tasks = [(s, seed) for s in strategies for seed in range(n_drafts)]

    t0 = time.perf_counter()
    with MPPool(max(os.cpu_count() - 1, 1), _init, (pool, actual)) as mp:
        rows = mp.map(_one_draft, tasks, chunksize=4)
    print(f"[{league_key}] {len(tasks)} drafts in "
          f"{time.perf_counter() - t0:.0f}s (N={n_drafts}, MC K={MC_K} R={MC_R})")

    df = pd.DataFrame(rows)
    df.to_csv(f"backtest_results_{league_key}.csv", index=False)

    summary = df.groupby("strategy").agg(
        mean_actual_points=("actual_points", "mean"),
        mean_rank=("rank", "mean"),
        pct_first=("rank", lambda r: 100 * (r == 1).mean()),
        pct_top3=("rank", lambda r: 100 * (r <= 3).mean()),
    ).round(2).loc[strategies]
    print("\n", summary.to_string())

    by_slot = df.pivot_table(index="slot", columns="strategy",
                             values="rank", aggfunc="mean").round(2)
    print("\nMean finishing rank by draft slot:\n", by_slot.to_string())

    mc, base = summary.loc["mc"], summary.loc["best_available"]
    print(f"\nHEADLINE [{league_key}]: In {n_drafts} simulated {label} drafts "
          f"scored week-by-week with real {label} results, our Monte Carlo "
          f"draft bot finished top-3 in {mc['pct_top3']:.0f}% of "
          f"{cfg['teams']}-team leagues (vs {base['pct_top3']:.0f}% for "
          f"drafting best-projected-available), with a mean finish of "
          f"{mc['mean_rank']:.1f} out of {cfg['teams']}.")


if __name__ == "__main__":
    main()
