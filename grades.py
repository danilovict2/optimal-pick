"""Percentile-based letter grades for picks and whole drafts.

Reference distributions come from simulated all-bot drafts: the average
projection of the player taken at each overall pick number, and the roster
value distribution per draft slot.
"""

import numpy as np

from draft_sim import run_draft, team_value

LETTERS = [(90, "A+"), (80, "A"), (70, "A-"), (60, "B+"), (50, "B"),
           (40, "B-"), (30, "C+"), (20, "C"), (10, "D"), (0, "F")]


def to_letter(pctl: float) -> str:
    for cut, letter in LETTERS:
        if pctl >= cut:
            return letter
    return "F"


def simulate_reference(pool, n_drafts: int = 200, seed: int = 7) -> dict:
    """Simulate n all-VOR-bot drafts. Returns per-pick-number mean/std of the
    drafted player's projection, and per-slot roster-value samples."""
    league = pool["league"]
    n_picks = league["teams"] * league["rounds"]
    taken = np.zeros((n_drafts, n_picks))
    slot_values = np.zeros((n_drafts, league["teams"]))
    for d in range(n_drafts):
        rosters, picks = run_draft(pool, seed=seed + d)
        taken[d] = [pool["proj"][i] for _, _, i in picks]
        slot_values[d] = [team_value(pool, ro) for ro in rosters]
    return {
        "pick_mean": taken.mean(axis=0),
        "pick_std": np.maximum(taken.std(axis=0), 1.0),
        "slot_values": slot_values,
    }


def pick_grade(pool, player_idx: int, pick_no: int, ref: dict):
    """Blend of (a) projection percentile within position among the pool and
    (b) value vs the average player taken at this pick in simulated drafts.
    pick_no is 0-based overall pick. Returns (letter, pctl, vs_pick_pts)."""
    proj = pool["proj"][player_idx]
    same_pos = pool["proj"][pool["pos_codes"] == pool["pos_codes"][player_idx]]
    pos_pctl = 100 * (same_pos < proj).mean()

    z = (proj - ref["pick_mean"][pick_no]) / ref["pick_std"][pick_no]
    value_pctl = 100 / (1 + np.exp(-1.2 * z))  # logistic squash of z-score

    pctl = 0.5 * pos_pctl + 0.5 * value_pctl
    return to_letter(pctl), pctl, proj - ref["pick_mean"][pick_no]


def draft_grade(pool, our_roster: list[int], slot: int, ref: dict):
    """Percentile of our roster value vs simulated drafts from the same slot.
    slot is 0-based. Returns (letter, pctl, our_value)."""
    val = team_value(pool, our_roster)
    dist = ref["slot_values"][:, slot]
    pctl = 100 * (dist < val).mean()
    return to_letter(pctl), pctl, val


if __name__ == "__main__":
    from config import LEAGUES
    from draft_sim import make_pool
    from projections import league_engine

    cfg = LEAGUES["nfl"]
    _, pool_df, _ = league_engine("nfl", backtest_season=cfg["backtest_season"])
    pool = make_pool(pool_df, cfg)

    ref = simulate_reference(pool, n_drafts=100)
    rosters, picks = run_draft(pool, seed=3, our_team=5, our_strategy="vor")
    print("Grades for team 6's picks (seed 3):")
    for n, (r, t, i) in enumerate(picks):
        if t == 5:
            letter, pctl, vs = pick_grade(pool, i, n, ref)
            print(f"  R{r+1:2d} pick {n+1:3d}: {pool['names'][i]:25s} "
                  f"{letter:2s} (pctl {pctl:4.0f}, {vs:+6.1f} vs avg pick)")
    letter, pctl, val = draft_grade(pool, rosters[5], 5, ref)
    print(f"Draft grade: {letter} (value {val:.0f}, pctl {pctl:.0f})")
