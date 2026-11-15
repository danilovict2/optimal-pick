"""Fantasy Premier League data: past seasons from the vaastav/
Fantasy-Premier-League GitHub archive, the current season from the official
FPL API.

player_id is the FPL global player code (stable across seasons). Points are
the official FPL points; positions GKP/DEF/MID/FWD. Seasons are labeled by
end year (2026 = 2025-26).
"""

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd

FIRST_SEASON = 2020  # the 2019-20 season

POSITIONS = ["GKP", "DEF", "MID", "FWD"]
ETYPE = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}  # 5 = manager chip: dropped
STATS = ["minutes", "goals_scored", "assists", "clean_sheets", "bonus"]
LAG_STATS = ["fantasy_points"] + STATS + ["games", "ppg"]

VAASTAV = ("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
           "master/data/{folder}/{file}")
API = "https://fantasy.premierleague.com/api/bootstrap-static/"
UA = {"User-Agent": "Mozilla/5.0"}  # the FPL API rejects default UAs

CACHE = Path(__file__).parent / "data_cache"


def season_label(season: int) -> str:
    return f"{season - 1}-{str(season)[2:]}"


def _current_season() -> int:
    t = date.today()
    return t.year + 1 if t.month >= 7 else t.year


def _bootstrap() -> dict:
    """Official bootstrap-static payload, cached daily."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / "fpl_bootstrap.json"
    if f.exists() and time.time() - f.stat().st_mtime < 86400:
        return json.loads(f.read_text())
    try:
        import requests
        d = requests.get(API, headers=UA, timeout=30).json()
        f.write_text(json.dumps(d))
        return d
    except Exception:
        if f.exists():
            return json.loads(f.read_text())
        raise


def _read_csv_cached(kind: str, season: int, file: str) -> pd.DataFrame:
    """Parquet-cached CSV from the vaastav archive. Past seasons never
    change; the current season's files refresh daily."""
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{kind}_{season}.parquet"
    final = season < _current_season()
    if f.exists() and (final or time.time() - f.stat().st_mtime < 86400):
        return pd.read_parquet(f)
    url = VAASTAV.format(folder=season_label(season), file=file)
    try:
        import io

        import requests
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), on_bad_lines="skip")
    except Exception:
        if f.exists():
            return pd.read_parquet(f)
        return pd.DataFrame()
    if len(df):
        df.to_parquet(f)
    return df


def _season_players(season: int) -> pd.DataFrame:
    """One row per player for a season, from the archive or (current
    season) the official API."""
    if season == _current_season():
        df = pd.DataFrame(_bootstrap()["elements"])
    else:
        df = _read_csv_cached("fpl_players", season, "players_raw.csv")
    if not len(df):
        return pd.DataFrame()
    out = pd.DataFrame({
        "player_id": df["code"].astype(str),
        "player_display_name":
            (df["first_name"].fillna("") + " "
             + df["second_name"].fillna("")).str.strip(),
        "position": df["element_type"].map(ETYPE),
        "season": season,
        "fantasy_points": pd.to_numeric(df["total_points"], errors="coerce"),
    })
    for c in STATS:
        out[c] = (pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                  if c in df else 0.0)
    games = (df["starts"] if "starts" in df
             else df["minutes"].astype(float) // 90)
    out["games"] = pd.to_numeric(games, errors="coerce").fillna(0).astype(int)
    out["ppg"] = out["fantasy_points"] / out["games"].clip(lower=1)
    return out.dropna(subset=["position", "fantasy_points"])


# ---------- league data contract ----------

def load_player_seasons() -> pd.DataFrame:
    frames = [_season_players(y)
              for y in range(FIRST_SEASON, _current_season() + 1)]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def load_weekly_points(season: int) -> pd.DataFrame:
    """player_id, week, fantasy_points from the season's merged gameweek
    log (double gameweeks sum)."""
    gw = _read_csv_cached("fpl_gw", season, "gws/merged_gw.csv")
    pr = _read_csv_cached("fpl_players", season, "players_raw.csv")
    if not len(gw) or not len(pr):
        raise RuntimeError(f"no FPL gameweek data for {season_label(season)}")
    codes = pr.set_index("id")["code"]
    week_col = "GW" if "GW" in gw else "round"
    out = pd.DataFrame({
        "player_id": gw["element"].map(codes).astype("Int64").astype(str),
        "week": pd.to_numeric(gw[week_col], errors="coerce"),
        "fantasy_points": pd.to_numeric(gw["total_points"], errors="coerce"),
    }).dropna()
    return out.groupby(["player_id", "week"], as_index=False)[
        "fantasy_points"].sum()


STATUS = {"a": "", "d": "Q", "i": "Injured", "s": "Suspended",
          "u": "Unavailable", "n": "Not available"}


def current_form():
    """Start/sit inputs: (form_df, period_label, note). form_df columns:
    player_id, name, position, form, proj, status."""
    d = _bootstrap()
    df = pd.DataFrame(d["elements"])
    df = df[df["element_type"].isin(ETYPE)]
    events = [e for e in d["events"] if e["finished"]]
    next_gw = (max(e["id"] for e in events) + 1) if events else 1

    form = pd.DataFrame({
        "player_id": df["code"].astype(str),
        "name": (df["first_name"].fillna("") + " "
                 + df["second_name"].fillna("")).str.strip(),
        "position": df["element_type"].map(ETYPE),
        "form": pd.to_numeric(df["form"], errors="coerce").fillna(0.0),
        "status": df["status"].map(STATUS).fillna(""),
    })
    out_flags = ("Injured", "Suspended", "Unavailable", "Not available")
    form["proj"] = form["form"].where(~form["status"].isin(out_flags), 0.0)
    note = ("Form = official FPL form (points per match, last 30 days); "
            "availability from the official status flags "
            "(Q = doubtful, flagged but started).")
    return form, f"Gameweek {next_gw}", note


if __name__ == "__main__":
    ps = load_player_seasons()
    print("player-seasons:", ps.shape, "seasons:",
          sorted(ps["season"].unique().tolist()))
    last = int(ps["season"].max())
    print(f"\nTop 5 {season_label(last)} by points:")
    print(ps[ps["season"] == last].nlargest(5, "fantasy_points")[
        ["player_display_name", "position", "games", "fantasy_points", "ppg"]
    ].round(1).to_string(index=False))
    wk = load_weekly_points(2026)
    print(f"\n2025-26 gameweek rows: {wk.shape}, weeks "
          f"{int(wk['week'].min())}-{int(wk['week'].max())}")
