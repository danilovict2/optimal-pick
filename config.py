"""League configurations. The engine only knows about players
(id, name, position, projected points) plus one of these configs.

Keys:
  label            display name
  module           data module exposing the league data contract
  slots            fixed starting slots {pos: count}
  flex             flex slots {name: (eligible positions, count)}
  max_starters     optional per-position cap on the whole starting lineup
                   (fixed + flex), e.g. FPL's formation limits
  teams, rounds    draft shape
  complete_games   latest season counts as finished once someone hit this
  backtest_season  the pinned historical-validation season
  max_per_pos      draft-bot roster caps
  one_until_round  bots don't take a 2nd of this position before round N
  not_before_round bots don't take this position at all before round N
  squad            exact roster quotas (FPL Draft enforces these)
"""

LEAGUES = {
    "nfl": {
        "label": "NFL",
        "module": "data_nfl",
        "slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        "flex": {"FLEX": (["RB", "WR", "TE"], 1)},
        "teams": 12,
        "rounds": 15,
        "complete_games": 15,
        "backtest_season": 2025,
        "max_per_pos": {"QB": 3, "TE": 3, "K": 1, "DST": 1},
        "one_until_round": {"QB": 8, "TE": 8},
        "not_before_round": {"K": 13, "DST": 12},
    },
    "nba": {
        "label": "NBA",
        "module": "data_nba",
        # ESPN's lineup folded onto the G/F/C positions the NBA publishes:
        # PG+SG+G = 3 G, SF+PF+F = 3 F, C, 3 UTIL, 3 bench
        "slots": {"G": 3, "F": 3, "C": 1},
        "flex": {"UTIL": (["G", "F", "C"], 3)},
        "teams": 10,
        "rounds": 13,
        "complete_games": 70,
        "backtest_season": 2026,  # the 2025-26 season
        "max_per_pos": {"G": 7, "F": 7, "C": 3},
    },
    "fpl": {
        "label": "Premier League",
        "module": "data_fpl",
        # FPL Draft: XI is 1 GKP + 3-5 DEF + 2-5 MID + 1-3 FWD
        "slots": {"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1},
        "flex": {"XI": (["DEF", "MID", "FWD"], 4)},
        "max_starters": {"DEF": 5, "MID": 5, "FWD": 3},
        "teams": 8,
        "rounds": 15,
        "complete_games": 30,
        "backtest_season": 2026,  # the 2025-26 season
        "squad": {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
    },
}
