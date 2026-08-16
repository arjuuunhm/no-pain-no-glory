"""The player-week spine: every RB/WR/TE roster-week, played or not.

This is the structural fix that the rest of the pipeline hangs off. The
previous version keyed the feature table off `weekly_player_stats`, which
only has a row when a player *recorded a stat line*. That made the table
outcome-conditioned in two ways:

1. **Availability was unmodelable.** A player who missed week 7 simply had no
   week-7 row, so "did this player play?" had no negative class to learn from
   -- the exact quantity resources.md §4 step 1 puts first.
2. **Team attribution came from the stat line**, so a player's team was only
   known on weeks he produced. Team-level joins (Vegas, O-line, scheme)
   therefore silently dropped every inactive week.

Building from `load_rosters_weekly()` instead gives one row per player per
week he was on an NFL roster, whether or not he took a snap. Playing status
becomes a *column* (and a label), not a filter.

Scoped to SKILL_POSITIONS (RB/WR/TE) and regular-season weeks. That drops
~72% of rows versus the old table -- which carried CB/LB/DE/K/P/OT rows that
no downstream model was ever going to use -- and makes every subsequent
groupby cheaper.

**Which roster statuses count.** Not all of them, and getting this wrong
quietly destroys the availability model. `load_rosters_weekly()` also carries
practice-squad (`DEV`) and released (`CUT`) rows: 30k of them for skill
positions in 2016-2024, at a measured 0.0% play rate. Left in, they would
make up 40% of the spine and nearly 60% of the negative class, and a model
would score beautifully by learning "practice squad => did not play" while
learning nothing about the question actually being asked. `ACTIVE_STATUSES`
restricts the spine to players under contract to the active roster, where
"did not play" means something -- inactive, injured reserve, PUP, suspended.
Those are missed games; a practice-squad week is not a missed game.
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import SKILL_POSITIONS, normalize_team

# Statuses where the player is on the active roster and *could* have played,
# so not playing is an informative event. Measured play rates 2016-2024:
# ACT 87.8%, everything else here ~0% (i.e. a genuine missed game).
# Deliberately excludes DEV (practice squad), CUT (released), and the
# low-volume transaction codes (RET/NWT/EXE/TRC/TRD/E14/RFA).
ACTIVE_STATUSES = ("ACT", "INA", "RES", "PUP", "SUS")


def build_preseason_spine(rosters: pl.DataFrame, season: int) -> pl.DataFrame:
    """A synthetic week-1 spine for a season that has not been played yet.

    `load_rosters_weekly()` does not carry a season before it starts -- it
    refuses 2026 outright -- so a draft-day projection for an upcoming season
    has no row to hang features off. The *seasonal* roster file does exist for
    the upcoming season, carries the same `status` vocabulary as the weekly
    one, and is the league's current answer to "who is on this team".

    Emitting it as week 1 is what makes the rest of the pipeline work
    unchanged: every feature block is keyed on (season, week) and already
    lagged, so a week-1 row of season N picks up season N-1 information and
    nothing else -- which is the whole definition of the draft-day information
    set (see `model/panel.py`).

    Two honest caveats, both about *when* this is built rather than how:

    - `roster_status` is nearly constant before final cuts. In the training
      seasons a week-1 row is ~81% ACT, because cuts and injured-reserve
      designations have happened by then; a mid-August roster is ~99% ACT.
      It is retained here only to define the active-roster spine; the learned
      projection deliberately does not use it as a feature.
    - `played` is False and `offense_snaps` 0 for every row, because no game
      has been played. These rows must never reach a label table.
    """
    spine = (
        rosters.filter(
            (pl.col("season") == season)
            & pl.col("position").is_in(SKILL_POSITIONS)
            & pl.col("status").is_in(ACTIVE_STATUSES)
            & pl.col("gsis_id").is_not_null()
        )
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("season").cast(pl.Int32),
            pl.lit(1, dtype=pl.Int32).alias("week"),
            "team",
            "position",
            pl.col("status").alias("roster_status"),
            pl.col("full_name").alias("player_name"),
        )
        .with_columns(normalize_team("team"))
        .sort(["player_id", "roster_status"])
        .unique(subset=["player_id"], keep="first", maintain_order=True)
    )
    return spine.with_columns(
        pl.lit(0, dtype=pl.Int64).alias("offense_snaps"),
        pl.lit(False).alias("played"),
    )


def build_spine(rosters_weekly: pl.DataFrame, snap_counts: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """One row per (player_id, season, week) for rostered RB/WR/TE in REG weeks."""
    spine = (
        rosters_weekly.filter(
            (pl.col("game_type") == "REG")
            & pl.col("position").is_in(SKILL_POSITIONS)
            & pl.col("status").is_in(ACTIVE_STATUSES)
            & pl.col("gsis_id").is_not_null()
        )
        .select(
            pl.col("gsis_id").alias("player_id"),
            "season",
            "week",
            "team",
            "position",
            pl.col("status").alias("roster_status"),
            pl.col("full_name").alias("player_name"),
        )
        .with_columns(normalize_team("team"))
        # A mid-week transaction can land a player on two rows; keep one.
        # Sorting by roster_status puts 'ACT' first alphabetically among the
        # codes that co-occur, so the active row wins.
        .sort(["player_id", "season", "week", "roster_status"])
        .unique(subset=["player_id", "season", "week"], keep="first", maintain_order=True)
    )

    # Realized participation, from offensive snaps rather than roster status:
    # 'ACT' means dressed, not that the player saw the field. snap_counts is
    # keyed by PFR id, so it comes in through the players.parquet crosswalk.
    crosswalk = (
        players.select(["gsis_id", "pfr_id"])
        .drop_nulls()
        .unique(subset=["pfr_id"])
        .rename({"gsis_id": "player_id"})
    )
    played = (
        snap_counts.select(["pfr_player_id", "season", "week", "offense_snaps"])
        .join(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")
        .group_by(["player_id", "season", "week"])
        .agg(pl.col("offense_snaps").max().alias("offense_snaps"))
    )

    spine = spine.join(played, on=["player_id", "season", "week"], how="left")
    return spine.with_columns(
        pl.col("offense_snaps").fill_null(0),
        # Absent from the snap report for a week he was rostered = did not play
        # an offensive snap that week.
        (pl.col("offense_snaps").fill_null(0) > 0).alias("played"),
    )
