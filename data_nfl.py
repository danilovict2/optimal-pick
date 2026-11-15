"""Load NFL weekly stats via nflreadpy and aggregate to player-season rows.

Covers all official-app roster positions: QB/RB/WR/TE/K come from player
stats (kicker fantasy points computed from FG distances since nflverse
leaves them at 0), and team defenses (DST) are built from team stats plus
schedule scores under standard D/ST scoring.
"""

import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

FIRST_SEASON = 2019

POSITIONS = ["QB", "RB", "WR", "TE", "K", "DST"]
PLAYER_POSITIONS = ["QB", "RB", "WR", "TE", "K"]  # rows in player stats

# usage/production stats summed over the season
SUM_STATS = [
    "fantasy_points",
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

LAG_STATS = SUM_STATS + MEAN_STATS + ["games", "ppg"]

CACHE = Path(__file__).parent / "data_cache"

# standard kicker scoring: FG by distance (0-39: 3, 40-49: 4, 50+: 5),
# PAT made: 1, FG missed: -1
_FG_MADE_3 = ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39"]
_FG_MADE_5 = ["fg_made_50_59", "fg_made_60_"]


def _kicker_points(df: pd.DataFrame) -> pd.Series:
    k = df.reindex(columns=_FG_MADE_3 + _FG_MADE_5
                   + ["fg_made_40_49", "fg_missed", "pat_made"]).fillna(0.0)
    return (3 * k[_FG_MADE_3].sum(axis=1) + 4 * k["fg_made_40_49"]
            + 5 * k[_FG_MADE_5].sum(axis=1) + k["pat_made"] - k["fg_missed"])


def _points_allowed_pts(pa: np.ndarray) -> np.ndarray:
    """Standard D/ST points-allowed tiers."""
    tiers = [(pa == 0, 10), (pa <= 6, 7), (pa <= 13, 4), (pa <= 20, 1),
             (pa <= 27, 0), (pa <= 34, -1)]
    return np.select([c for c, _ in tiers], [v for _, v in tiers], default=-4)


def _dst_weekly(year: int) -> pd.DataFrame:
    """One row per team-week with standard D/ST fantasy scoring: sacks 1,
    INT 2, fumble recovery 2, defensive/special-teams TD 6, safety 2,
    blocked kick 2, plus points-allowed tiers from final scores."""
    import nflreadpy as nfl
    ts = nfl.load_team_stats([year]).to_pandas()
    ts = ts[ts["season_type"] == "REG"].copy()
    sch = nfl.load_schedules([year]).to_pandas()
    sch = sch[sch["game_type"] == "REG"]
    pa = pd.concat([
        sch.rename(columns={"home_team": "team", "away_score": "pa"})[
            ["week", "team", "pa"]],
        sch.rename(columns={"away_team": "team", "home_score": "pa"})[
            ["week", "team", "pa"]],
    ])
    ts = ts.merge(pa, on=["team", "week"], how="left")
    d = ts.reindex(columns=[
        "def_sacks", "def_interceptions", "fumble_recovery_opp", "def_tds",
        "special_teams_tds", "def_safeties", "def_punt_blocks",
        "def_pat_blocks", "def_fg_blocks"]).fillna(0.0)
    pts = (d["def_sacks"] + 2 * d["def_interceptions"]
           + 2 * d["fumble_recovery_opp"]
           + 6 * (d["def_tds"] + d["special_teams_tds"])
           + 2 * d["def_safeties"]
           + 2 * (d["def_punt_blocks"] + d["def_pat_blocks"]
                  + d["def_fg_blocks"])
           + _points_allowed_pts(ts["pa"].fillna(0.0).to_numpy(float)))
    out = pd.DataFrame({
        "player_id": "DST_" + ts["team"],
        "player_display_name": ts["team"] + " D/ST",
        "position": "DST",
        "team": ts["team"],
        "season": ts["season"].astype(int),
        "week": ts["week"].astype(int),
        "fantasy_points": pts.to_numpy(float),
    })
    for c in SUM_STATS[1:] + MEAN_STATS:
        out[c] = 0.0
    return out


def load_weekly(seasons: list[int] | None = None) -> pd.DataFrame:
    """Weekly regular-season stats for QB/RB/WR/TE/K/DST, cached per season
    as parquet. seasons=None loads everything from FIRST_SEASON through the
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
        f = CACHE / f"weekly_{y}_v3.parquet"
        final = y < this_year  # past seasons never change
        if f.exists() and (final or time.time() - f.stat().st_mtime < 86400):
            frames.append(pd.read_parquet(f).rename(
                columns={"fantasy_points_ppr": "fantasy_points"}))
            continue
        import nflreadpy as nfl
        try:
            df = nfl.load_player_stats([y]).to_pandas()
            df = df[(df["season_type"] == "REG")
                    & (df["position"].isin(PLAYER_POSITIONS))].copy()
            df = df.rename(columns={"fantasy_points_ppr": "fantasy_points"})
            # nflverse leaves kicker fantasy points at 0: compute them
            kick = df["position"] == "K"
            df.loc[kick, "fantasy_points"] = _kicker_points(df[kick])
            df = pd.concat([df[keep], _dst_weekly(y)[keep]], ignore_index=True)
        except Exception:
            if f.exists():  # offline/not-yet-published: fall back to cache
                frames.append(pd.read_parquet(f))
            continue
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
    out["ppg"] = out["fantasy_points"] / out["games"]
    return out


# ---------- league data contract ----------

def season_label(season: int) -> str:
    return str(season)


def load_player_seasons() -> pd.DataFrame:
    return player_seasons(load_weekly())


def load_weekly_points(season: int) -> pd.DataFrame:
    """player_id, week, fantasy_points for one season (backtest scoring)."""
    w = load_weekly([season])
    return w[["player_id", "week", "fantasy_points"]]


def current_form():
    """Start/sit inputs: (form_df, period_label, note). form_df columns:
    player_id, name, position, form, proj, status."""
    weekly = load_weekly()
    latest = int(weekly["season"].max())
    recent = weekly[weekly["season"] == latest].sort_values("week")
    last_done = int(recent["week"].max())
    this_week = last_done + 1

    form = (recent.groupby("player_id")
            .agg(name=("player_display_name", "last"),
                 position=("position", "last"),
                 team=("team", "last"),
                 form=("fantasy_points", lambda s: s.tail(4).mean()),
                 last_gw=("week", "max"))
            .reset_index())

    inj = load_injuries(latest)
    inj_week = None
    if len(inj):
        inj_week = min(this_week, int(inj["week"].max()))
        inj = (inj[inj["week"] == inj_week]
               .drop_duplicates("gsis_id").set_index("gsis_id")["report_status"])
    else:
        inj = pd.Series(dtype=object)
    byes = bye_teams(latest, this_week)

    def _status(r):
        if r["team"] in byes:
            return "BYE", 0.0
        s = inj.get(r["player_id"])
        if s in ("Out", "Doubtful"):
            return s, 0.0
        if s == "Questionable":
            return "Q", r["form"]
        if r["last_gw"] < last_done - 2:  # missed the last 3+ weeks, no status
            return f"no game since wk {int(r['last_gw'])}", 0.0
        return "", r["form"]

    form[["status", "proj"]] = form.apply(_status, axis=1, result_type="expand")
    inj_note = (f"injury report wk {inj_week}" if inj_week is not None
                else "no injury report available")
    bye_note = ", ".join(sorted(byes)) if byes else "none"
    note = (f"Form = last 4 games through week {last_done}; {inj_note}; "
            f"week {this_week} byes: {bye_note}.")
    return form, f"Week {this_week}", note


if __name__ == "__main__":
    weekly = load_weekly()
    print("weekly:", weekly.shape, "seasons:",
          sorted(weekly["season"].unique().tolist()))
    ps = player_seasons(weekly)
    print("player-seasons:", ps.shape)
    last = int(ps["season"].max())
    print(ps[ps["season"] == last].nlargest(5, "fantasy_points")[
        ["player_display_name", "position", "games", "fantasy_points", "ppg"]
    ].to_string(index=False))
