"""Saved leagues: a finished draft becomes a JSON file under leagues/."""

import json
import re
from datetime import date
from pathlib import Path

LEAGUES_DIR = Path(__file__).parent / "leagues"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", name.lower().replace(" ", "-")) or "league"


def save_league(name: str, league: dict, draft_season: int, my_slot: int,
                rosters: list[list[dict]], live: bool,
                league_key: str = "nfl") -> str:
    """rosters: per team, a list of {player_id, name, position, proj} dicts.
    Returns the final (possibly de-duplicated) league name."""
    LEAGUES_DIR.mkdir(exist_ok=True)
    final = name
    n = 2
    while (LEAGUES_DIR / f"{_slug(final)}.json").exists():
        final = f"{name} ({n})"
        n += 1
    data = {
        "name": final,
        "league": league_key,
        "created": date.today().isoformat(),
        "draft_season": draft_season,
        "teams": league["teams"],
        "rounds": league["rounds"],
        "my_slot": my_slot,  # 0-based
        "live": live,
        "rosters": rosters,
    }
    (LEAGUES_DIR / f"{_slug(final)}.json").write_text(json.dumps(data, indent=1))
    return final


def list_leagues(league_key: str | None = None) -> list[str]:
    """Saved league names, newest first; league_key filters by sport
    (files from before the multi-league era count as NFL)."""
    if not LEAGUES_DIR.exists():
        return []
    files = sorted(LEAGUES_DIR.glob("*.json"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    out = []
    for f in files:
        data = json.loads(f.read_text())
        if league_key is None or data.get("league", "nfl") == league_key:
            out.append(data["name"])
    return out


def load_league(name: str) -> dict:
    return json.loads((LEAGUES_DIR / f"{_slug(name)}.json").read_text())
