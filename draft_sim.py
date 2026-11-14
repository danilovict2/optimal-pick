"""Snake draft simulator: opponent bots, baseline bots, Monte Carlo bot.

Everything in the inner loop is numpy on a flat player pool:
  proj[i], pos_codes[i] (index into pos_order), vor[i], available mask,
  per-team position counts. No pandas inside simulations.
"""

import numpy as np
import pandas as pd

from optimizer import roster_value

MAX_PER_POS = {"QB": 3, "TE": 3}       # roster caps for bots
ONE_UNTIL_ROUND = {"QB": 8, "TE": 8}   # no 2nd QB/TE before this round (1-based)


def make_pool(pool_df: pd.DataFrame, league: dict) -> dict:
    """Precompute numpy arrays from a projection pool DataFrame
    (columns: player_id, name, position, proj), sorted by proj desc."""
    df = pool_df.sort_values("proj", ascending=False).reset_index(drop=True)
    pos_order = list(league["slots"].keys())
    proj = df["proj"].to_numpy(float)
    pos_codes = df["position"].map({p: i for i, p in enumerate(pos_order)}).to_numpy()

    # replacement level: Nth best at pos, N = teams * (fixed starters + even
    # share of each flex slot the position is eligible for)
    teams = league["teams"]
    starters = {p: float(n) for p, n in league["slots"].items()}
    for _, (elig, n) in league["flex"].items():
        for p in elig:
            starters[p] += n / len(elig)
    repl = np.zeros(len(pos_order))
    for ci, p in enumerate(pos_order):
        pp = np.sort(proj[pos_codes == ci])[::-1]
        n = min(int(round(teams * starters[p])), len(pp)) - 1
        repl[ci] = pp[max(n, 0)]
    vor = proj - repl[pos_codes]
    noise_std = np.maximum(0.1 * np.abs(vor), 1.0)

    return {
        "df": df, "names": df["name"].tolist(), "proj": proj,
        "pos_codes": pos_codes, "vor": vor, "noise_std": noise_std,
        "pos_order": pos_order, "league": league,
    }


def snake_order(teams: int, rounds: int):
    order = []
    for r in range(rounds):
        row = range(teams) if r % 2 == 0 else range(teams - 1, -1, -1)
        order += [(r, t) for t in row]
    return order


def _eligible(pool, avail, pos_counts_team, round_no):
    """Mask of players this team may draft (roster caps)."""
    mask = avail.copy()
    for p, cap in MAX_PER_POS.items():
        ci = pool["pos_order"].index(p)
        if pos_counts_team[ci] >= cap or (
            pos_counts_team[ci] >= 1 and round_no + 1 < ONE_UNTIL_ROUND[p]
        ):
            mask &= pool["pos_codes"] != ci
    return mask if mask.any() else avail


def vor_pick(pool, rng, avail, pos_counts_team, round_no):
    """Opponent bot: VOR + gaussian noise, with roster caps."""
    mask = _eligible(pool, avail, pos_counts_team, round_no)
    idx = np.flatnonzero(mask)
    scores = pool["vor"][idx] + rng.normal(0, pool["noise_std"][idx])
    return idx[np.argmax(scores)]


def best_available_pick(pool, rng, avail, pos_counts_team, round_no):
    """Baseline: highest projection available (same sanity caps)."""
    mask = _eligible(pool, avail, pos_counts_team, round_no)
    idx = np.flatnonzero(mask)
    return idx[np.argmax(pool["proj"][idx])]


def team_value(pool, roster: list[int], bench_weight: float = 0.1) -> float:
    r = np.asarray(roster, dtype=int)
    return roster_value(pool["proj"][r], pool["pos_codes"][r], pool["league"],
                        pool["pos_order"], bench_weight)


def _rollout(pool, rng, avail, pos_counts, our_roster, our_team, order, start,
             bench_weight):
    """Simulate picks order[start:] with the VOR bot policy for everyone
    (including our future picks); return our final roster value."""
    for r, t in order[start:]:
        i = vor_pick(pool, rng, avail, pos_counts[t], r)
        avail[i] = False
        pos_counts[t][pool["pos_codes"][i]] += 1
        if t == our_team:
            our_roster.append(i)
    return team_value(pool, our_roster, bench_weight)


def mc_pick(pool, rng, avail, pos_counts, our_roster, our_team, order, pick_no,
            K=8, R=64, bench_weight=0.1):
    """Monte Carlo bot: evaluate top-K VOR candidates by mean roster value
    over R rollouts. Returns (pick, candidates list for the 'why' table)."""
    r_now = order[pick_no][0]
    mask = _eligible(pool, avail, pos_counts[our_team], r_now)
    idx = np.flatnonzero(mask)
    cands = idx[np.argsort(pool["vor"][idx])[::-1][:K]]

    results = []
    for c in cands:
        vals = np.empty(R)
        for rep in range(R):
            a = avail.copy()
            pc = pos_counts.copy()
            a[c] = False
            pc[our_team][pool["pos_codes"][c]] += 1
            vals[rep] = _rollout(pool, rng, a, pc, our_roster + [int(c)],
                                 our_team, order, pick_no + 1, bench_weight)
        results.append({"idx": int(c), "name": pool["names"][c],
                        "position": pool["pos_order"][pool["pos_codes"][c]],
                        "proj": float(pool["proj"][c]),
                        "mean_value": float(vals.mean())})
    results.sort(key=lambda d: -d["mean_value"])
    return results[0]["idx"], results


def run_draft(pool, seed, our_team=None, our_strategy="vor", K=8, R=64,
              bench_weight=0.1):
    """Run one full draft. our_strategy: 'vor' | 'best_available' | 'mc'.
    Returns (rosters, picks) where picks is [(round, team, player_idx)]."""
    rng = np.random.default_rng(seed)
    league = pool["league"]
    order = snake_order(league["teams"], league["rounds"])
    avail = np.ones(len(pool["proj"]), dtype=bool)
    pos_counts = np.zeros((league["teams"], len(pool["pos_order"])), dtype=int)
    rosters = [[] for _ in range(league["teams"])]
    picks = []
    for n, (r, t) in enumerate(order):
        if t == our_team and our_strategy == "mc":
            i, _ = mc_pick(pool, rng, avail, pos_counts, rosters[t], t, order,
                           n, K, R, bench_weight)
        elif t == our_team and our_strategy == "best_available":
            i = best_available_pick(pool, rng, avail, pos_counts[t], r)
        else:
            i = vor_pick(pool, rng, avail, pos_counts[t], r)
        avail[i] = False
        pos_counts[t][pool["pos_codes"][i]] += 1
        rosters[t].append(int(i))
        picks.append((r, t, int(i)))
    return rosters, picks


if __name__ == "__main__":
    import time
    from config import LEAGUES
    from data_nfl import load_weekly, player_seasons
    from projections import train_model, project

    weekly = load_weekly(list(range(2019, 2026)))
    ps = player_seasons(weekly)
    model, cols = train_model(ps, [2022, 2023, 2024])
    pool = make_pool(project(ps, model, cols, 2025), LEAGUES["nfl"])

    rosters, picks = run_draft(pool, seed=42)
    print("Rounds 1-3 board (12 VOR bots):")
    for r, t, i in picks[:36]:
        print(f"  R{r+1} T{t+1:2d}: {pool['names'][i]:28s} "
              f"{pool['pos_order'][pool['pos_codes'][i]]:3s} {pool['proj'][i]:6.1f}")
    vals = [team_value(pool, ro) for ro in rosters]
    print("team values:", np.round(vals, 1))

    # MC timing check: our first pick from slot 5 (pick_no 4)
    rng = np.random.default_rng(0)
    order = snake_order(12, 15)
    avail = np.ones(len(pool["proj"]), dtype=bool)
    pos_counts = np.zeros((12, 4), dtype=int)
    for n in range(4):
        r, t = order[n]
        i = vor_pick(pool, rng, avail, pos_counts[t], r)
        avail[i] = False
        pos_counts[t][pool["pos_codes"][i]] += 1
    t0 = time.perf_counter()
    i, cands = mc_pick(pool, rng, avail, pos_counts, [], 4, order, 4, K=8, R=64)
    dt = time.perf_counter() - t0
    print(f"\nMC pick (K=8, R=64) took {dt:.2f}s -> {pool['names'][i]}")
    for c in cands:
        print(f"  {c['name']:25s} {c['position']:3s} proj {c['proj']:6.1f} "
              f"mean value {c['mean_value']:7.1f}")
