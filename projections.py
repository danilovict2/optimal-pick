"""Feature building + model + projected season fantasy points.

Features for target season y use only seasons < y (lag-1 and lag-2 stats).
Rookies (no prior NFL season) get a low position-specific default: the 30th
percentile of that position's model projections.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from data_nfl import POSITIONS, SUM_STATS, MEAN_STATS, load_weekly, player_seasons

LAG_STATS = SUM_STATS + MEAN_STATS + ["games", "ppg"]
ROOKIE_PCTL = 30


def season_setup(ps: pd.DataFrame) -> tuple[list[int], int]:
    """Decide what to train on and what to project from whatever data exists.
    The latest season counts as complete once someone has played 15+ games;
    a complete latest season means the next draft is for the following year.
    Returns (train_target_seasons, draft_season)."""
    seasons = sorted(int(s) for s in ps["season"].unique())
    last = seasons[-1]
    complete = ps.loc[ps["season"] == last, "games"].max() >= 15
    draft_season = last + 1 if complete else last
    # targets need lag features: skip the first two loaded seasons
    train_targets = [y for y in seasons if y >= seasons[0] + 2 and y < draft_season]
    return train_targets, draft_season


def build_features(ps: pd.DataFrame, target_seasons: list[int]) -> pd.DataFrame:
    """One row per (player, target season y) for players with a season y-1.
    Columns: lag1_*/lag2_* stats, position one-hots, and y (actual points,
    NaN when the player has no season-y row)."""
    rows = []
    for y in target_seasons:
        lag1 = ps[ps["season"] == y - 1].set_index("player_id")
        lag2 = ps[ps["season"] == y - 2].set_index("player_id")
        actual = ps[ps["season"] == y].set_index("player_id")["fantasy_points_ppr"]
        f = lag1[LAG_STATS].add_prefix("lag1_")
        f = f.join(lag2[LAG_STATS].add_prefix("lag2_"))
        f["position"] = lag1["position"]
        f["player_display_name"] = lag1["player_display_name"]
        f["target_season"] = y
        f["y"] = actual
        rows.append(f.reset_index())
    df = pd.concat(rows, ignore_index=True)
    for pos in POSITIONS:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)
    return df


FEATURE_COLS = None  # set on first train


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("lag1_", "lag2_", "pos_"))]


def train_model(ps: pd.DataFrame, train_targets: list[int]):
    df = build_features(ps, train_targets).dropna(subset=["y"])
    cols = _feature_cols(df)
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(df[cols], df["y"])
    return model, cols


def project(ps: pd.DataFrame, model, cols, target_season: int) -> pd.DataFrame:
    """Projection pool for `target_season`: veterans (played y-1) get model
    projections; rookies (appear in y with no prior season) get the position's
    30th-percentile projection. Returns player_id, name, position, proj."""
    vets = build_features(ps, [target_season])
    vets["proj"] = model.predict(vets[cols]).clip(min=0)
    vets = vets[["player_id", "player_display_name", "position", "proj"]]

    prior_ids = set(ps.loc[ps["season"] < target_season, "player_id"])
    cur = ps[ps["season"] == target_season]
    rookies = cur[~cur["player_id"].isin(prior_ids)][
        ["player_id", "player_display_name", "position"]
    ].copy()
    defaults = vets.groupby("position")["proj"].quantile(ROOKIE_PCTL / 100)
    rookies["proj"] = rookies["position"].map(defaults)

    pool = pd.concat([vets, rookies], ignore_index=True)
    return pool.rename(columns={"player_display_name": "name"}).sort_values(
        "proj", ascending=False
    ).reset_index(drop=True)


if __name__ == "__main__":
    weekly = load_weekly(list(range(2019, 2026)))
    ps = player_seasons(weekly)

    # validation: train on 2022-2023 targets, evaluate on held-out 2024
    model_v, cols = train_model(ps, [2022, 2023])
    val = build_features(ps, [2024]).dropna(subset=["y"])
    val["pred"] = model_v.predict(val[cols])
    print("Validation MAE on held-out 2024 (season-total PPR points):")
    for pos in POSITIONS:
        sub = val[val["position"] == pos]
        print(f"  {pos}: {mean_absolute_error(sub['y'], sub['pred']):6.1f}  (n={len(sub)})")

    # backtest model: train on 2022-2024 targets, project 2025
    model, cols = train_model(ps, [2022, 2023, 2024])
    pool = project(ps, model, cols, 2025)
    print(f"\n2025 projection pool: {len(pool)} players")
    for pos in POSITIONS:
        top = pool[pool["position"] == pos].head(24)
        print(f"\nTop 24 {pos} projected for 2025:")
        print(top[["name", "proj"]].to_string(index=False))
