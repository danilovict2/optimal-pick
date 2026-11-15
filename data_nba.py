"""NBA data via nba_api (stats.nba.com): per-game logs aggregated to
player-season rows.

Fantasy scoring is Yahoo-style points: PTS + 1.2*REB + 1.5*AST + 3*STL +
3*BLK - TOV. Positions are the G/F/C the NBA publishes (hybrids like G-F
take their first letter). Seasons are labeled by end year (2025 = 2024-25).
"""

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_SEASON = 2020  # the 2019-20 season

POSITIONS = ["G", "F", "C"]
STATS = ["pts", "reb", "ast", "stl", "blk", "tov", "min"]
LAG_STATS = ["fantasy_points"] + STATS + ["games", "ppg"]

CACHE = Path(__file__).parent / "data_cache"


def season_label(season: int) -> str:
    return f"{season - 1}-{str(season)[2:]}"


def _current_season() -> int:
    t = date.today()
    return t.year + 1 if t.month >= 10 else t.year


def _fetch_logs(season: int) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs
    df = playergamelogs.PlayerGameLogs(
        season_nullable=season_label(season),
        season_type_nullable="Regular Season", timeout=90,
    ).get_data_frames()[0]
    df = df[["PLAYER_ID", "PLAYER_NAME", "GAME_DATE",
             "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN"]].copy()
    df.columns = ["player_id", "player_display_name", "game_date"] + STATS
    df["player_id"] = df["player_id"].astype(str)
    df["season"] = season
    df["fantasy_points"] = (df["pts"] + 1.2 * df["reb"] + 1.5 * df["ast"]
                            + 3 * df["stl"] + 3 * df["blk"] - df["tov"])
    return df


def load_games(seasons: list[int] | None = None) -> pd.DataFrame:
    """Per-player per-game rows, cached per season as parquet. Past seasons
    are cached forever; the current season refreshes daily."""
    CACHE.mkdir(exist_ok=True)
    cur = _current_season()
    if seasons is None:
        seasons = list(range(FIRST_SEASON, cur + 1))
    frames = []
    for y in seasons:
        f = CACHE / f"nba_games_{y}.parquet"
        final = y < cur
        if f.exists() and (final or time.time() - f.stat().st_mtime < 86400):
            frames.append(pd.read_parquet(f))
            continue
        try:
            df = _fetch_logs(y)
        except Exception:
            if f.exists():
                frames.append(pd.read_parquet(f))
            continue
        if len(df):
            df.to_parquet(f)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _positions() -> pd.Series:
    """player_id -> G/F/C from the all-history player index (cached)."""
    f = CACHE / "nba_positions.parquet"
    if f.exists() and time.time() - f.stat().st_mtime < 7 * 86400:
        df = pd.read_parquet(f)
    else:
        try:
            from nba_api.stats.endpoints import playerindex
            df = playerindex.PlayerIndex(
                season=season_label(_current_season()),
                historical_nullable=1, timeout=90,
            ).get_data_frames()[0][["PERSON_ID", "POSITION"]]
            df.columns = ["player_id", "position"]
            df["player_id"] = df["player_id"].astype(str)
            df.to_parquet(f)
        except Exception:
            if not f.exists():
                raise
            df = pd.read_parquet(f)
    pos = df.set_index("player_id")["position"].astype(str).str[0]
    return pos[pos.isin(POSITIONS)]


# ---------- league data contract ----------

def load_player_seasons() -> pd.DataFrame:
    g = load_games()
    grp = g.groupby(["player_id", "season"])
    agg = grp[["fantasy_points"] + STATS].sum()
    agg["games"] = grp["game_date"].count()
    agg = agg.reset_index()
    names = grp["player_display_name"].last().reset_index()
    out = agg.merge(names, on=["player_id", "season"])
    out["ppg"] = out["fantasy_points"] / out["games"]
    out["position"] = out["player_id"].map(_positions())
    return out.dropna(subset=["position"])


def load_weekly_points(season: int) -> pd.DataFrame:
    """player_id, week, fantasy_points. Fantasy weeks run Monday-Sunday
    from the week of the season's first game."""
    g = load_games([season])
    d = pd.to_datetime(g["game_date"])
    start = d.min().normalize()
    start -= pd.Timedelta(days=start.weekday())  # back to that Monday
    g = g.assign(week=((d - start).dt.days // 7) + 1)
    return (g.groupby(["player_id", "week"], as_index=False)
            ["fantasy_points"].sum())


def current_form():
    """Start/sit inputs: (form_df, period_label, note). form_df columns:
    player_id, name, position, form, proj, status."""
    g = load_games([_current_season()]).sort_values("game_date")
    last_date = pd.to_datetime(g["game_date"]).max()
    form = (g.groupby("player_id")
            .agg(name=("player_display_name", "last"),
                 form=("fantasy_points", lambda s: s.tail(4).mean()),
                 last_date=("game_date", "max"))
            .reset_index())
    form["position"] = form["player_id"].map(_positions())
    form = form.dropna(subset=["position"])
    stale = pd.to_datetime(form["last_date"]) < last_date - timedelta(days=10)
    form["status"] = np.where(stale, "no recent games", "")
    form["proj"] = np.where(stale, 0.0, form["form"])
    note = (f"Form = last 4 games through {last_date.date()}. Players "
            f"without a game in 10+ days are treated as out.")
    return (form[["player_id", "name", "position", "form", "proj", "status"]],
            "This week", note)


if __name__ == "__main__":
    ps = load_player_seasons()
    print("player-seasons:", ps.shape, "seasons:",
          sorted(ps["season"].unique().tolist()))
    last = int(ps["season"].max())
    print(f"\nTop 5 {season_label(last)} by fantasy points:")
    print(ps[ps["season"] == last].nlargest(5, "fantasy_points")[
        ["player_display_name", "position", "games", "fantasy_points", "ppg"]
    ].round(1).to_string(index=False))
