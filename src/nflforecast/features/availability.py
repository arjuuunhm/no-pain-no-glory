"""Availability features -- resources.md §4 step 1, "will he be on the field".

The decomposition puts availability first for a reason: for a season-long
projection it is the largest single source of variance, and unlike efficiency
it is partly predictable. This module supplies the *features* for that model.
The label it predicts lives in labels.py.

**This is the half of the pipeline the rework changed most.** The previous
version computed a player-season `games_played` and joined it into the
player-week table on `(player_id, season)`, which broadcast a whole-season
outcome across every week of that season -- week 3 of 2023 carried how many
games the player would go on to play through week 18 of 2023. It was
described in its own docstring as a label, and it was, but nothing stopped it
being read as a feature, and any model that touched it would have looked
extraordinary in validation for entirely fake reasons.

What is here now is strictly backward-looking:

- `prev_season_availability_rate` / `prev_season_games_played` -- the
  player's *completed* prior season, attached by a season+1 join key.
  Durability has real year-over-year signal; this is how to use it.
- `played_rate_last{3,5,8}` -- recent participation, shift(1)-lagged over the
  roster spine, so it reads the weeks before this one only.
- `games_missed_season_to_date` -- absences so far this season, again
  excluding the current week.
- `injury_report_status` / `injury_practice_status` / `depth_chart_rank` --
  the current week's pre-game report. Safe at the week-N cutoff because all
  three are published before kickoff. `load_injuries()` is team-self-reported
  and known to be gamed (docs/features.md §6), so it is exposed as a raw
  categorical for the model to discount, never treated as ground truth.

Grain is the spine's: one row per rostered RB/WR/TE per regular-season week,
including weeks the player did not take a snap. Those rows are the entire
point -- they are the positive class for "missed a game".
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import ROLLING_WINDOWS
from nflforecast.features.utils import add_trailing_rolling

# Regular-season game count by season: 16 games through 2020, 17 from 2021 on.
REGULAR_SEASON_EXPANSION_YEAR = 2021


def _scheduled_games() -> pl.Expr:
    return pl.when(pl.col("season") >= REGULAR_SEASON_EXPANSION_YEAR).then(17).otherwise(16)


def build_prior_season_availability(spine: pl.DataFrame) -> pl.DataFrame:
    """Completed-prior-season durability, keyed to the following season."""
    season_level = (
        spine.group_by(["player_id", "season"])
        .agg(
            pl.col("played").sum().alias("prev_season_games_played"),
            pl.len().alias("prev_season_roster_weeks"),
        )
        .with_columns(
            (pl.col("prev_season_games_played") / _scheduled_games()).alias(
                "prev_season_availability_rate"
            )
        )
        # Shift onto the next season via the key, not shift(), so a player who
        # was out of the league for a year cannot pick up a stale row.
        .with_columns((pl.col("season") + 1).alias("season"))
    )
    return season_level


def build_participation_history(spine: pl.DataFrame) -> pl.DataFrame:
    """Trailing participation over the roster spine, lagged one week."""
    df = spine.select(["player_id", "season", "week", "played"]).with_columns(
        pl.col("played").cast(pl.Int8)
    )

    df = add_trailing_rolling(
        df, group_col="player_id", order_cols=["season", "week"],
        value_cols=["played"], windows=list(ROLLING_WINDOWS),
    ).rename({f"played_last{w}": f"played_rate_last{w}" for w in ROLLING_WINDOWS})

    # Season-to-date absences, excluding the current week.
    df = df.sort(["player_id", "season", "week"]).with_columns(
        (1 - pl.col("played"))
        .shift(1)
        .cum_sum()
        .over(["player_id", "season"])
        .fill_null(0)
        .alias("games_missed_season_to_date")
    )

    return df.drop("played")


def build_injury_report_features(injuries: pl.DataFrame, depth_charts: pl.DataFrame) -> pl.DataFrame:
    # One row per player per week: keep the last-dated report as the pre-game
    # snapshot (status can move Wed -> Fri; see docs/features.md §7).
    inj = (
        injuries.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
        .sort(["gsis_id", "season", "week", "date_modified"])
        .group_by(["gsis_id", "season", "week"], maintain_order=True)
        .last()
        .select(
            pl.col("gsis_id").alias("player_id"),
            "season",
            "week",
            pl.col("report_status").alias("injury_report_status"),
            pl.col("practice_status").alias("injury_practice_status"),
        )
    )

    dc = (
        depth_charts.select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("season").cast(pl.Int32),
            pl.col("week").cast(pl.Int32),
            pl.col("depth_team").alias("depth_chart_rank"),
        )
        .filter(pl.col("player_id").is_not_null())
        .unique(subset=["player_id", "season", "week"], keep="first")
    )

    return inj.join(dc, on=["player_id", "season", "week"], how="full", coalesce=True)


def fill_injury_report_defaults(df: pl.DataFrame) -> pl.DataFrame:
    """Absence from the injury report means no listed issue, not missing data.

    Applied after the join onto the spine rather than inside
    build_injury_report_features: ~95% of skill-position player-weeks have no
    report at all, and those nulls only become interpretable once they sit on
    a row we know describes a rostered player.
    """
    return df.with_columns(
        pl.col("injury_report_status").fill_null("Healthy"),
        pl.col("injury_practice_status").fill_null("Healthy"),
    )
