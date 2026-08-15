"""Opportunity / volume features -- the core of the model (resources.md §4 step 2).

This is the block that matters most. Volume shares are the metrics that
actually stabilise year over year, so this is where boosting earns its keep;
efficiency is regressed toward positional means elsewhere rather than
modelled hard.

Per player-game we take target share, air-yards share, WOPR, carries,
targets, receptions and snap share, then express each as a trailing average
over the player's last 3/5/8 *games played* plus a season-to-date mean.

**Leakage control.** Nothing here is a feature at its own week. Rolling is
computed inclusive of each played game, and `attach_asof` (see utils.py) then
binds each spine week to the most recent row strictly before it. Two
consequences worth stating, because they are the point of the rework:

- A player's week-9 features come from games <= week 8, never week 9.
- A player who *missed* week 9 still gets week-9 features -- his form as of
  week 8 -- so the availability model has something to score. Under the old
  shift-then-roll-on-stat-lines approach those rows did not exist at all.

Raw current-week columns are not emitted. They live in labels.py, which is
the only place they belong; keeping them beside the features was a standing
invitation to train on the outcome.

snap_counts is keyed by pfr_player_id (PFR), not gsis_id like every other
table -- joined here through players.parquet's gsis_id<->pfr_id crosswalk.
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import ROLLING_WINDOWS, SKILL_POSITIONS
from nflforecast.features.utils import add_inclusive_rolling, add_season_to_date

VALUE_COLS = [
    "target_share",
    "air_yards_share",
    "wopr",
    "carries",
    "targets",
    "receptions",
    "offense_pct",
]


def build_opportunity_features(
    weekly_stats: pl.DataFrame,
    snap_counts: pl.DataFrame,
    players: pl.DataFrame,
) -> pl.DataFrame:
    crosswalk = (
        players.select(["gsis_id", "pfr_id"])
        .drop_nulls()
        .unique(subset=["pfr_id"])
    )

    snaps = (
        snap_counts.join(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .select(["gsis_id", "season", "week", "offense_pct"])
        .rename({"gsis_id": "player_id"})
        .group_by(["player_id", "season", "week"])
        .agg(pl.col("offense_pct").max().alias("offense_pct"))
    )

    df = (
        weekly_stats.filter(
            (pl.col("season_type") == "REG") & pl.col("position").is_in(SKILL_POSITIONS)
        )
        .select(
            ["player_id", "season", "week"]
            + [c for c in VALUE_COLS if c != "offense_pct"]
        )
        .join(snaps, on=["player_id", "season", "week"], how="left")
    )

    df = add_inclusive_rolling(
        df, group_col="player_id", order_cols=["season", "week"],
        value_cols=VALUE_COLS, windows=list(ROLLING_WINDOWS),
    )
    df = add_season_to_date(
        df, group_col="player_id", season_col="season",
        order_cols=["week"], value_cols=VALUE_COLS,
    )

    # Raw per-week values are the outcome, not a feature -- drop them here and
    # rebuild them in labels.py where the intent is explicit.
    return df.drop(VALUE_COLS)
