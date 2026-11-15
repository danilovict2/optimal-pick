"""Snake draft simulator: opponent bots, baseline bots, Monte Carlo bot.

Everything in the inner loop is numpy on a flat player pool:
  proj[i], pos_codes[i] (index into pos_order), vor[i], available mask,
  per-team position counts. No pandas inside simulations.
"""

import numpy as np
import pandas as pd

from optimizer import roster_value

# bot behavior (roster caps, timing rules, squad quotas) lives in the
# league config -- see config.py


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
    """Mask of players this team may draft (roster caps + must-fill)."""
    pos_order, league = pool["pos_order"], pool["league"]
    quota = league.get("squad") or league["slots"]

    # must-fill: once remaining picks are only just enough to fill every
    # unmet quota slot (starting slots, or FPL's exact squad), draft those
    picks_left = league["rounds"] - round_no
    need = {p: n - pos_counts_team[pos_order.index(p)]
            for p, n in quota.items()
            if pos_counts_team[pos_order.index(p)] < n}
    if need and sum(need.values()) >= picks_left:
        codes = [pos_order.index(p) for p in need]
        mask = avail & np.isin(pool["pos_codes"], codes)
        if mask.any():
            return mask

    # exact squad quotas double as hard caps
    caps = {**league.get("max_per_pos", {}), **league.get("squad", {})}
    one_until = league.get("one_until_round", {})
    mask = avail.copy()
    for p, cap in caps.items():
        if p not in pos_order:
            continue
        ci = pos_order.index(p)
        if pos_counts_team[ci] >= cap or (
            pos_counts_team[ci] >= 1
            and round_no + 1 < one_until.get(p, 0)
        ):
            mask &= pool["pos_codes"] != ci
    for p, rnd in league.get("not_before_round", {}).items():
        if p in pos_order and round_no + 1 < rnd:
            mask &= pool["pos_codes"] != pos_order.index(p)
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


def _candidates(pool, idx, our_roster, K, bench_weight):
    """Top-K candidates, half by VOR (scarcity) and half by marginal value
    added to OUR current roster (so a 3rd RB ranks below a needed starter),
    interleaved and deduped."""
    by_vor = idx[np.argsort(pool["vor"][idx])[::-1]]
    r = np.asarray(our_roster, dtype=int)
    rp, rc = pool["proj"][r], pool["pos_codes"][r]
    league, pos_order = pool["league"], pool["pos_order"]
    base = (roster_value(rp, rc, league, pos_order, bench_weight)
            if len(r) else 0.0)
    # shortlist for the marginal scores: deep VOR cut + each position's best
    short = set(by_vor[:4 * K].tolist())
    for ci in range(len(pos_order)):
        pos_idx = idx[pool["pos_codes"][idx] == ci]
        if len(pos_idx):
            short.add(int(pos_idx[np.argmax(pool["proj"][pos_idx])]))
    short = np.fromiter(short, dtype=int)
    marg = np.array([
        roster_value(np.append(rp, pool["proj"][i]),
                     np.append(rc, pool["pos_codes"][i]),
                     league, pos_order, bench_weight) - base
        for i in short])
    by_marg = short[np.argsort(marg)[::-1]]
    merged = []
    for pair in zip(by_vor[:K], by_marg[:K]):
        merged += [int(pair[0]), int(pair[1])]
    return list(dict.fromkeys(merged))[:K]


def mc_pick(pool, rng, avail, pos_counts, our_roster, our_team, order, pick_no,
            K=8, R=64, bench_weight=0.1):
    """Monte Carlo bot: evaluate top-K candidates (VOR + roster-need mix)
    by mean roster value over R rollouts. Returns (pick, candidates list
    for the 'why' table)."""
    r_now = order[pick_no][0]
    mask = _eligible(pool, avail, pos_counts[our_team], r_now)
    idx = np.flatnonzero(mask)
    cands = _candidates(pool, idx, our_roster, K, bench_weight)

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
    import sys
    import time

    from config import LEAGUES
    from projections import league_engine

    league_key = sys.argv[1] if len(sys.argv) > 1 else "nfl"
    cfg = LEAGUES[league_key]
    _, pool_df, _ = league_engine(league_key,
                                  backtest_season=cfg["backtest_season"])
    pool = make_pool(pool_df, cfg)

    rosters, picks = run_draft(pool, seed=42)
    print(f"[{league_key}] Rounds 1-3 board ({cfg['teams']} VOR bots):")
    for r, t, i in picks[:cfg["teams"] * 3]:
        print(f"  R{r+1} T{t+1:2d}: {pool['names'][i]:28s} "
              f"{pool['pos_order'][pool['pos_codes'][i]]:3s} {pool['proj'][i]:6.1f}")
    vals = [team_value(pool, ro) for ro in rosters]
    print("team values:", np.round(vals, 1))

    # MC timing check: our first pick from slot 5 (pick_no 4)
    rng = np.random.default_rng(0)
    order = snake_order(cfg["teams"], cfg["rounds"])
    avail = np.ones(len(pool["proj"]), dtype=bool)
    pos_counts = np.zeros((cfg["teams"], len(pool["pos_order"])), dtype=int)
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
