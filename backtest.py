"""Backtest on the real 2025 season: train on <=2024, draft as if August 2025,
then score every team week by week: for each of the 18 regular-season weeks,
the roster's best legal lineup from that week's actual PPR points (so byes,
mid-season injuries, and weekly variance all count), summed over the season.

Runs N drafts per strategy (our seat: Monte Carlo / best-available / VOR bot;
other 11 seats: VOR bots). MC uses R=16 rollouts here (vs 64 in the app) to
keep runtime down; drafts run in parallel across CPU cores.
"""

import os
import time
from multiprocessing import Pool as MPPool

import numpy as np
import pandas as pd

from config import LEAGUES
from data_nfl import load_weekly, player_seasons
from draft_sim import make_pool, run_draft
from optimizer import greedy_lineup_value
from projections import draft_pool, train_model, project

N_DRAFTS = 1000
MC_R = 16
MC_K = 8
POOL_TOP = 400  # bots never draft past ~pick 250; truncate pool for speed

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


def build_pool_and_actuals():
    weekly = load_weekly(list(range(2019, 2026)))
    ps = player_seasons(weekly)
    model, cols = train_model(ps, [2022, 2023, 2024])  # no 2025 leakage
    pool_df = draft_pool(project(ps, model, cols, 2025), POOL_TOP)
    pool = make_pool(pool_df, LEAGUES["nfl"])

    # (players, weeks) matrix of actual 2025 weekly points, pool-aligned;
    # byes and missed games are simply 0 that week
    wk = weekly[weekly["season"] == 2025]
    mat = wk.pivot_table(index="player_id", columns="week",
                         values="fantasy_points_ppr", aggfunc="sum")
    actual = mat.reindex(pool["df"]["player_id"]).fillna(0.0).to_numpy(float)
    return pool, actual


def main():
    pool, actual = build_pool_and_actuals()
    strategies = ["mc", "best_available", "vor"]
    tasks = [(s, seed) for s in strategies for seed in range(N_DRAFTS)]

    t0 = time.perf_counter()
    with MPPool(max(os.cpu_count() - 1, 1), _init, (pool, actual)) as mp:
        rows = mp.map(_one_draft, tasks, chunksize=4)
    print(f"{len(tasks)} drafts in {time.perf_counter() - t0:.0f}s "
          f"(N={N_DRAFTS}, MC K={MC_K} R={MC_R})")

    df = pd.DataFrame(rows)
    df.to_csv("backtest_results.csv", index=False)

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
    print(f"\nHEADLINE: In {N_DRAFTS} simulated 2025 drafts scored week-by-week "
          f"with real 2025 results, our Monte Carlo draft bot finished top-3 in "
          f"{mc['pct_top3']:.0f}% of 12-team leagues (vs {base['pct_top3']:.0f}% "
          f"for drafting best-projected-available), with a mean finish of "
          f"{mc['mean_rank']:.1f} out of 12.")


if __name__ == "__main__":
    main()
