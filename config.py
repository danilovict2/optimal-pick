"""League configurations. The engine only knows about players
(id, name, position, projected points) plus one of these configs."""

LEAGUES = {
    "nfl": {
        "slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        "flex": {"FLEX": (["RB", "WR", "TE"], 1)},
        "teams": 12,
        "rounds": 15,
    },
    # "nba": {...}  # stretch goal
}
