"""Feature building + model + projected season fantasy points.

League-agnostic: each league's data module supplies POSITIONS, LAG_STATS and
load_player_seasons(), and every player-season frame shares the contract
columns player_id, season, player_display_name, position, fantasy_points,
games, ppg.

Features for target season y use only seasons < y (lag-1 and lag-2 stats).
Newcomers (no prior season in the league -- NFL/NBA rookies, FPL signings)
get a low position-specific default: the 30th percentile of that position's
model projections.
"""

import importlib

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from config import LEAGUES

ROOKIE_PCTL = 30


def league_module(league_key: str):
    return importlib.import_module(LEAGUES[league_key]["module"])


def season_setup(ps: pd.DataFrame, complete_games: int) -> tuple[list[int], int]:
    """Decide what to train on and what to project from whatever data exists.
    The latest season counts as complete once someone has played
    complete_games; a complete latest season means the next draft is for the
    following year. Returns (train_target_seasons, draft_season)."""
    seasons = sorted(int(s) for s in ps["season"].unique())
    last = seasons[-1]
    complete = ps.loc[ps["season"] == last, "games"].max() >= complete_games
    draft_season = last + 1 if complete else last
    # targets need lag features: skip the first two loaded seasons
    train_targets = [y for y in seasons if y >= seasons[0] + 2 and y < draft_season]
    return train_targets, draft_season


def build_features(ps: pd.DataFrame, target_seasons: list[int],
                   positions: list[str], lag_stats: list[str]) -> pd.DataFrame:
    """One row per (player, target season y) for players with a season y-1.
    Columns: lag1_*/lag2_* stats, position one-hots, and y (actual points,
    NaN when the player has no season-y row)."""
    rows = []
    for y in target_seasons:
        lag1 = ps[ps["season"] == y - 1].set_index("player_id")
        lag2 = ps[ps["season"] == y - 2].set_index("player_id")
        actual = ps[ps["season"] == y].set_index("player_id")["fantasy_points"]
        f = lag1[lag_stats].add_prefix("lag1_")
        f = f.join(lag2[lag_stats].add_prefix("lag2_"))
        f["position"] = lag1["position"]
        f["player_display_name"] = lag1["player_display_name"]
        f["target_season"] = y
        f["y"] = actual
        rows.append(f.reset_index())
    df = pd.concat(rows, ignore_index=True)
    for pos in positions:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)
    return df


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("lag1_", "lag2_", "pos_"))]


def train_model(ps: pd.DataFrame, train_targets: list[int],
                positions: list[str], lag_stats: list[str]):
    df = build_features(ps, train_targets, positions, lag_stats).dropna(subset=["y"])
    cols = _feature_cols(df)
    model = HistGradientBoostingRegressor(random_state=0)
    model.fit(df[cols], df["y"])
    return model, cols


def project(ps: pd.DataFrame, model, cols, target_season: int,
            positions: list[str], lag_stats: list[str]) -> pd.DataFrame:
    """Projection pool for `target_season`: veterans (played y-1) get model
    projections; newcomers (appear in y with no prior season) get the
    position's 30th-percentile projection. Returns player_id, name,
    position, proj."""
    vets = build_features(ps, [target_season], positions, lag_stats)
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


def draft_pool(pool: pd.DataFrame, n: int = 400,
               min_per_pos: int = 20) -> pd.DataFrame:
    """Top-n projected players, but guarantee every position keeps at least
    min_per_pos entries so low-scoring slots (K, DST, GKP) stay draftable."""
    keep = pool.index < n
    for pos in pool["position"].unique():
        keep |= pool.index.isin(pool.index[pool["position"] == pos][:min_per_pos])
    return pool[keep].reset_index(drop=True)


def league_engine(league_key: str, backtest_season: int | None = None):
    """Load a league's data, train its model, and build its draft pool.
    backtest_season pins the projection target and trains strictly on
    earlier seasons. Returns (ps, pool_df, draft_season)."""
    cfg = LEAGUES[league_key]
    mod = league_module(league_key)
    ps = mod.load_player_seasons()
    seasons = sorted(int(s) for s in ps["season"].unique())
    if backtest_season is not None:
        draft_season = backtest_season
        targets = [y for y in seasons
                   if y >= seasons[0] + 2 and y < draft_season]
    else:
        targets, draft_season = season_setup(ps, cfg["complete_games"])
    model, cols = train_model(ps, targets, mod.POSITIONS, mod.LAG_STATS)
    pool_df = draft_pool(
        project(ps, model, cols, draft_season, mod.POSITIONS, mod.LAG_STATS))
    return ps, pool_df, draft_season


if __name__ == "__main__":
    import sys

    league_key = sys.argv[1] if len(sys.argv) > 1 else "nfl"
    cfg = LEAGUES[league_key]
    mod = league_module(league_key)
    ps = mod.load_player_seasons()
    seasons = sorted(int(s) for s in ps["season"].unique())
    _, draft_season = season_setup(ps, cfg["complete_games"])
    bt = cfg["backtest_season"]

    # validation: hold out the season before the backtest season
    val_season = bt - 1
    val_targets = [y for y in seasons if y >= seasons[0] + 2 and y < val_season]
    model_v, cols = train_model(ps, val_targets, mod.POSITIONS, mod.LAG_STATS)
    val = build_features(ps, [val_season], mod.POSITIONS,
                         mod.LAG_STATS).dropna(subset=["y"])
    val["pred"] = model_v.predict(val[cols])
    print(f"[{league_key}] validation MAE on held-out "
          f"{mod.season_label(val_season)} (season-total points):")
    for pos in mod.POSITIONS:
        sub = val[val["position"] == pos]
        if len(sub):
            print(f"  {pos}: {mean_absolute_error(sub['y'], sub['pred']):6.1f}"
                  f"  (n={len(sub)})")

    _, pool, _ = league_engine(league_key, backtest_season=bt)
    print(f"\n{mod.season_label(bt)} projection pool: {len(pool)} players")
    for pos in mod.POSITIONS:
        top = pool[pool["position"] == pos].head(10)
        print(f"\nTop 10 {pos} projected for {mod.season_label(bt)}:")
        print(top[["name", "proj"]].round(1).to_string(index=False))
