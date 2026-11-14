"""ILP lineup optimizer (source of truth) + fast greedy evaluator."""

import numpy as np
import pandas as pd
import pulp


def _slot_list(league: dict) -> list[tuple[str, list[str], int]]:
    """[(slot_name, eligible_positions, count), ...]"""
    slots = [(pos, [pos], n) for pos, n in league["slots"].items()]
    slots += [(name, elig, n) for name, (elig, n) in league["flex"].items()]
    return slots


def best_lineup(players_df: pd.DataFrame, league: dict):
    """ILP assignment: maximize total projected points of the starting lineup.
    players_df needs columns: name, position, proj. Returns (total, lineup_df)."""
    df = players_df.reset_index(drop=True)
    slots = _slot_list(league)
    prob = pulp.LpProblem("lineup", pulp.LpMaximize)
    x = {}
    for i, pos in enumerate(df["position"]):
        for s, (sname, elig, _) in enumerate(slots):
            if pos in elig:
                x[i, s] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")
    prob += pulp.lpSum(df.loc[i, "proj"] * v for (i, s), v in x.items())
    for s, (sname, elig, count) in enumerate(slots):
        vars_s = [v for (i, s2), v in x.items() if s2 == s]
        prob += pulp.lpSum(vars_s) == min(count, len(vars_s))
    for i in range(len(df)):
        vars_i = [v for (i2, s), v in x.items() if i2 == i]
        if vars_i:
            prob += pulp.lpSum(vars_i) <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    chosen = [(i, slots[s][0]) for (i, s), v in x.items() if v.value() == 1]
    lineup = df.loc[[i for i, _ in chosen]].copy()
    lineup["slot"] = [sl for _, sl in chosen]
    return lineup["proj"].sum(), lineup


def greedy_lineup_value(proj: np.ndarray, pos_codes: np.ndarray, league: dict,
                        pos_order: list[str]) -> float:
    """Fast greedy: fill fixed position slots with each position's best, then
    flex slots with best remaining eligible. proj must be the roster's
    projections, pos_codes the index of each player's position in pos_order."""
    used = np.zeros(len(proj), dtype=bool)
    total = 0.0
    for pos, n in league["slots"].items():
        p = pos_order.index(pos)
        idx = np.where((pos_codes == p) & ~used)[0]
        best = idx[np.argsort(proj[idx])[::-1][:n]]
        total += proj[best].sum()
        used[best] = True
    for _, (elig, n) in league["flex"].items():
        elig_codes = [pos_order.index(e) for e in elig]
        idx = np.where(np.isin(pos_codes, elig_codes) & ~used)[0]
        best = idx[np.argsort(proj[idx])[::-1][:n]]
        total += proj[best].sum()
        used[best] = True
    return float(total)


def roster_value(proj: np.ndarray, pos_codes: np.ndarray, league: dict,
                 pos_order: list[str], bench_weight: float = 0.1) -> float:
    """Starting-lineup value + bench_weight * sum of remaining players' proj."""
    starters = greedy_lineup_value(proj, pos_codes, league, pos_order)
    return starters + bench_weight * (proj.sum() - starters)


if __name__ == "__main__":
    from config import LEAGUES
    from data_nfl import POSITIONS

    league = LEAGUES["nfl"]
    rng = np.random.default_rng(0)
    mismatches = 0
    for t in range(200):
        n = rng.integers(8, 16)
        df = pd.DataFrame({
            "name": [f"p{i}" for i in range(n)],
            "position": rng.choice(POSITIONS, size=n, p=[0.15, 0.3, 0.4, 0.15]),
            "proj": np.round(rng.uniform(10, 300, size=n), 2),
        })
        ilp_total, _ = best_lineup(df, league)
        codes = np.array([POSITIONS.index(p) for p in df["position"]])
        g = greedy_lineup_value(df["proj"].to_numpy(), codes, league, POSITIONS)
        if abs(ilp_total - g) > 1e-6:
            mismatches += 1
            print(f"MISMATCH trial {t}: ILP={ilp_total:.2f} greedy={g:.2f}")
    print(f"greedy == ILP on {200 - mismatches}/200 random rosters")
