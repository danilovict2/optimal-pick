"""Load NFL weekly stats via nflreadpy and aggregate to player-season rows."""

import time
from datetime import date
from pathlib import Path

import pandas as pd

FIRST_SEASON = 2019

POSITIONS = ["QB", "RB", "WR", "TE"]

# usage/production stats summed over the season
SUM_STATS = [
    "fantasy_points_ppr",
    "targets",
    "receptions",
    "carries",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rushing_yards",
    "rushing_tds",
    "receiving_yards",
    "receiving_tds",
]
# per-game-ish rates averaged over the season
MEAN_STATS = ["target_share"]

CACHE = Path(__file__).parent / "data_cache"


def load_weekly(seasons: list[int] | None = None) -> pd.DataFrame:
    """Weekly regular-season stats for QB/RB/WR/TE, cached per season as
    parquet. seasons=None loads everything from FIRST_SEASON through the
    latest season nflreadpy has data for. Past seasons are cached forever;
    the current calendar year's season is re-downloaded daily."""
    CACHE.mkdir(exist_ok=True)
    this_year = date.today().year
    if seasons is None:
        seasons = list(range(FIRST_SEASON, this_year + 1))

    keep = (["player_id", "player_display_name", "position", "team", "season",
             "week"] + SUM_STATS + MEAN_STATS)
    frames = []
    for y in seasons:
        f = CACHE / f"weekly_{y}_v2.parquet"
        final = y < this_year  # past seasons never change
        if f.exists() and (final or time.time() - f.stat().st_mtime < 86400):
            frames.append(pd.read_parquet(f))
            continue
        import nflreadpy as nfl
        try:
            df = nfl.load_player_stats([y]).to_pandas()
        except Exception:
            if f.exists():  # offline/not-yet-published: fall back to cache
                frames.append(pd.read_parquet(f))
            continue
        df = df[(df["season_type"] == "REG") & (df["position"].isin(POSITIONS))]
        df = df[keep].copy()
        if len(df):
            df.to_parquet(f)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_cached(kind: str, season: int, fetch) -> pd.DataFrame:
    """Daily-refresh parquet cache for small per-season tables."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{kind}_{season}.parquet"
    final = season < date.today().year
    if f.exists() and (final or time.time() - f.stat().st_mtime < 86400):
        return pd.read_parquet(f)
    try:
        df = fetch()
    except Exception:
        return pd.read_parquet(f) if f.exists() else pd.DataFrame()
    if len(df):
        df.to_parquet(f)
    return df


def load_injuries(season: int) -> pd.DataFrame:
    """Weekly injury reports: gsis_id, week, report_status (Out/Doubtful/
    Questionable or NaN). Empty frame if unavailable."""
    def fetch():
        import nflreadpy as nfl
        df = nfl.load_injuries([season]).to_pandas()
        return df[["gsis_id", "week", "report_status"]].dropna(
            subset=["report_status"])
    return _load_cached("injuries", season, fetch)


def load_schedule(season: int) -> pd.DataFrame:
    """Game schedule: week, home_team, away_team."""
    def fetch():
        import nflreadpy as nfl
        df = nfl.load_schedules([season]).to_pandas()
        return df[df["game_type"] == "REG"][["week", "home_team", "away_team"]]
    return _load_cached("schedule", season, fetch)


def bye_teams(season: int, week: int) -> set[str]:
    """Teams with no scheduled game in the given week."""
    sched = load_schedule(season)
    if not len(sched) or week > sched["week"].max():
        return set()
    wk = sched[sched["week"] == week]
    all_teams = set(sched["home_team"]) | set(sched["away_team"])
    return all_teams - set(wk["home_team"]) - set(wk["away_team"])


def player_seasons(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly rows to one row per player-season."""
    grp = weekly.groupby(["player_id", "season"])
    agg = grp[SUM_STATS].sum()
    agg[MEAN_STATS] = grp[MEAN_STATS].mean()
    agg["games"] = grp["week"].nunique()
    agg = agg.reset_index()
    # take position/name from the most recent week of that season
    meta = weekly.sort_values("week").groupby(["player_id", "season"]).last()[
        ["player_display_name", "position"]
    ].reset_index()
    out = agg.merge(meta, on=["player_id", "season"])
    out["ppg"] = out["fantasy_points_ppr"] / out["games"]
    return out


if __name__ == "__main__":
    weekly = load_weekly()
    print("weekly:", weekly.shape, "seasons:",
          sorted(weekly["season"].unique().tolist()))
    ps = player_seasons(weekly)
    print("player-seasons:", ps.shape)
    last = int(ps["season"].max())
    print(ps[ps["season"] == last].nlargest(5, "fantasy_points_ppr")[
        ["player_display_name", "position", "games", "fantasy_points_ppr", "ppg"]
    ].to_string(index=False))
