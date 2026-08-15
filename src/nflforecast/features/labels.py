"""Prediction targets, kept deliberately separate from the feature table.

The features table and this table are written as two files that share the
`(player_id, season, week)` key. That separation is the enforcement mechanism
for everything the feature modules promise about leakage: a raw current-week
number cannot be accidentally trained on if it does not live in the file the
model reads.

Targets follow the resources.md §4 decomposition, one per stage:

1. **Availability** -- `played` (weekly, on the spine) and
   `season_games_played` / `season_availability_rate` (player-season).
2. **Opportunity per game** -- `target_share`, `carries`, `offense_pct` and
   friends, as realised that week. The main event: these are the quantities
   the middle stage predicts.
3. **Efficiency / points** -- `fantasy_points`, `fantasy_points_ppr`, plus
   the yardage and touchdown components, so efficiency can be scored on its
   own rather than only through the points total.

Season-level rows are emitted at season grain and are *not* broadcast across
weeks -- doing that is what made the old table leak.
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import SKILL_POSITIONS

WEEKLY_TARGET_COLS = [
    "fantasy_points",
    "fantasy_points_ppr",
    "target_share",
    "air_yards_share",
    "wopr",
    "targets",
    "receptions",
    "carries",
    "receiving_yards",
    "rushing_yards",
    "receiving_tds",
    "rushing_tds",
]


def build_weekly_labels(spine: pl.DataFrame, weekly_stats: pl.DataFrame) -> pl.DataFrame:
    """Per player-week targets, on the spine so missed games are explicit zeros."""
    stats = weekly_stats.filter(
        (pl.col("season_type") == "REG") & pl.col("position").is_in(SKILL_POSITIONS)
    ).select(["player_id", "season", "week", *WEEKLY_TARGET_COLS])

    df = spine.select(["player_id", "season", "week", "team", "position", "played"]).join(
        stats, on=["player_id", "season", "week"], how="left"
    )

    # A rostered player who did not play scored zero -- that is a real
    # observation, and treating it as missing would quietly restrict every
    # downstream model to the players who were healthy enough to appear.
    return df.with_columns(
        [
            pl.when(pl.col("played")).then(pl.col(c)).otherwise(0.0).alias(c)
            for c in WEEKLY_TARGET_COLS
        ]
    )


def build_season_labels(spine: pl.DataFrame, weekly_labels: pl.DataFrame) -> pl.DataFrame:
    """Player-season targets: availability plus per-game production rates."""
    from nflforecast.features.availability import _scheduled_games

    played = (
        spine.group_by(["player_id", "season"])
        .agg(
            pl.col("played").sum().alias("season_games_played"),
            pl.len().alias("season_roster_weeks"),
        )
        .with_columns(
            (pl.col("season_games_played") / _scheduled_games()).alias("season_availability_rate")
        )
    )

    totals = weekly_labels.group_by(["player_id", "season"]).agg(
        pl.col("fantasy_points_ppr").sum().alias("season_fantasy_points_ppr"),
        pl.col("fantasy_points").sum().alias("season_fantasy_points"),
        pl.col("targets").sum().alias("season_targets"),
        pl.col("carries").sum().alias("season_carries"),
    )

    df = played.join(totals, on=["player_id", "season"], how="left")
    return df.with_columns(
        # Per-game rates conditional on playing: the quantity a projection
        # multiplies by predicted games, so it must exclude the zeros above.
        pl.when(pl.col("season_games_played") > 0)
        .then(pl.col("season_fantasy_points_ppr") / pl.col("season_games_played"))
        .alias("season_ppr_per_game"),
    )
