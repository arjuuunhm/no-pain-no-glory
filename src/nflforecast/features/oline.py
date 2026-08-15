"""Offensive-line quality proxies, at team-week grain.

There is no free line-charting feed in nflverse -- no snap-level blocking
grades, no yards-before-contact. What play-by-play *does* support is the
family of box-score proxies Football Outsiders built its line metrics from,
which isolate the line by exploiting where in a play's outcome distribution
the line's contribution lives:

**Run blocking**
- `ol_adj_line_yards` -- Line Yards. Credits the line with 120% of yards on a
  stuffed run, 100% of yards 0-4, 50% of yards 5-10, and 0% beyond. The
  banding is the whole point: yards past ~10 are the back breaking tackles in
  space, not the line opening a hole, so capping them strips most of the
  running back's contribution out of the metric.

  Two deliberate departures from Football Outsiders' published metric, both
  of which make the number here *not* comparable to theirs:

  * **No opponent adjustment** (the "Adjusted" in Adjusted Line Yards) --
    that needs a defensive-front rating we do not build.
  * **No rescaling.** FO normalises so the league mean matches league yards
    per carry, which is why their number is ~4.2. The raw banded credit is
    ~2.7 on 2016-2024 (verified: mean designed-run yardage 4.11, mean banded
    credit 2.72), and that is what this column holds. Rescaling was rejected
    on purpose: it would divide by a league-season average, i.e. by a
    quantity computed from games that have not happened yet at the point the
    feature is used. A monotone transform buys a boosted tree nothing and
    would cost the leakage guarantee.
- `ol_stuff_rate` -- share of designed runs gaining <= 0 yards. The clearest
  single "the line lost immediately" signal.
- `ol_power_success` -- conversion rate on runs with <= 2 yards to go on
  3rd/4th down or at the goal line. Low-n per game (see rolling note below)
  but it is the situation where line push is least confounded by scheme.

**Pass protection**
- `ol_sack_rate`, `ol_qb_hit_rate` -- per dropback. Genuinely confounded by
  the quarterback: a QB who holds the ball inflates both regardless of his
  line. We expose them anyway because for our purposes the confound is
  mostly harmless -- what a receiver's projection cares about is whether
  dropbacks survive long enough to become targets, and the QB's contribution
  to that is part of the signal, not noise to be removed.

All of it is a *team* property, so a player inherits his team's row. Two
consequences the code has to respect:

- Everything is lagged. A team's week-N sack rate includes week N, so the
  raw column can never be a feature for week N; only the `_last{3,5,8}`
  columns leave this module, same rule as opportunity.py.
- Rolling is over team-weeks, so windows are stable in a way the per-player
  ones are not -- a team plays every week, so `_last3` really is the last
  three games rather than "last three games this player appeared in".
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import ROLLING_WINDOWS, normalize_team
from nflforecast.features.utils import add_trailing_rolling

# Football Outsiders Adjusted Line Yards banding.
STUFF_MULTIPLIER = 1.2
FULL_CREDIT_YARDS = 4  # yards 0-4 credited to the line in full
HALF_CREDIT_YARDS = 6  # yards 5-10 credited at 50%

OUTPUT_COLS = [
    "ol_adj_line_yards",
    "ol_stuff_rate",
    "ol_power_success",
    "ol_sack_rate",
    "ol_qb_hit_rate",
]


def _line_yard_credit() -> pl.Expr:
    y = pl.col("yards_gained")
    return (
        pl.when(y < 0)
        .then(y * STUFF_MULTIPLIER)
        .otherwise(
            pl.min_horizontal(y, pl.lit(FULL_CREDIT_YARDS))
            + (y - FULL_CREDIT_YARDS).clip(0, HALF_CREDIT_YARDS) * 0.5
        )
    )


def build_oline_features(pbp: pl.DataFrame) -> pl.DataFrame:
    reg = pbp.filter((pl.col("season_type") == "REG") & pl.col("posteam").is_not_null()).with_columns(
        normalize_team("posteam")
    )

    # Designed runs only -- scrambles are a dropback that broke down, and
    # counting them as run-blocking reps would credit the line for the QB
    # escaping bad protection.
    runs = reg.filter(
        (pl.col("rush_attempt") == 1)
        & (pl.col("qb_scramble") != 1)
        & pl.col("yards_gained").is_not_null()
    ).with_columns(
        _line_yard_credit().alias("_line_yards"),
        (pl.col("yards_gained") <= 0).alias("_stuffed"),
        (
            ((pl.col("ydstogo") <= 2) & pl.col("down").is_in([3, 4]))
            | ((pl.col("goal_to_go") == 1) & (pl.col("ydstogo") <= 2))
        ).alias("_power_situation"),
        (pl.col("first_down") == 1).alias("_converted"),
    )

    run_agg = runs.group_by(["posteam", "season", "week"]).agg(
        pl.col("_line_yards").mean().alias("ol_adj_line_yards"),
        pl.col("_stuffed").mean().alias("ol_stuff_rate"),
        # Null, not 0, when a team saw no power situation that week -- "never
        # tried" is not "tried and failed", and the rolling mean below skips
        # nulls rather than dragging the average down.
        pl.col("_converted").filter(pl.col("_power_situation")).mean().alias("ol_power_success"),
    )

    dropbacks = reg.filter(pl.col("qb_dropback") == 1)
    pass_agg = dropbacks.group_by(["posteam", "season", "week"]).agg(
        pl.col("sack").mean().alias("ol_sack_rate"),
        pl.col("qb_hit").mean().alias("ol_qb_hit_rate"),
    )

    df = (
        run_agg.join(pass_agg, on=["posteam", "season", "week"], how="full", coalesce=True)
        .rename({"posteam": "team"})
        .sort(["team", "season", "week"])
    )

    df = add_trailing_rolling(
        df, group_col="team", order_cols=["season", "week"],
        value_cols=OUTPUT_COLS, windows=list(ROLLING_WINDOWS),
    )
    # Raw current-week values describe the game being predicted; drop them so
    # they cannot be picked up as features by accident.
    return df.drop(OUTPUT_COLS)
