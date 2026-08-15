"""Player priors -- age and draft capital, v1 tier item 4 (docs/features.md §10).

Grain is player-season (not player-week): age moves slowly enough within a
season that a single as-of-season-start value is sufficient for v1, and
draft capital/experience are already season-scoped in load_rosters_weekly.

Per docs/features.md §5 / CLAUDE.md: age curves are strongly
position-specific and non-linear, and draft capital dominates the prior for
players with <2 NFL seasons. We expose raw age_years and draft_pick (not a
hand-picked transform) alongside position, and let the boosted model find
the position-specific shape from splits -- per docs/features.md §5, "there's
no strong evidence one functional form dominates ... let the GBM find the
shape, as long as raw pick number is exposed as a feature."

Undrafted players get draft_pick = 300 (beyond the last real pick, so a GBM
split at "very high pick number" naturally isolates them) plus an explicit
is_undrafted flag, rather than a null that could be silently mishandled.
"""

from __future__ import annotations

import polars as pl

UNDRAFTED_PICK_SENTINEL = 300
SEASON_START_MONTH_DAY = "-09-01"


def build_prior_features(rosters_weekly: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    season_grain = (
        rosters_weekly.group_by(["gsis_id", "season"])
        .agg(pl.col("years_exp").max().alias("years_exp"))
        .rename({"gsis_id": "player_id"})
    )

    bio = players.select(
        ["gsis_id", "birth_date", "draft_year", "draft_round", "draft_pick", "position"]
    ).rename({"gsis_id": "player_id"})

    df = season_grain.join(bio, on="player_id", how="left")

    df = df.with_columns(
        pl.col("birth_date").str.to_date(strict=False).alias("birth_date"),
        (pl.col("season").cast(pl.Utf8) + pl.lit(SEASON_START_MONTH_DAY))
        .str.to_date(strict=False)
        .alias("_season_start"),
    )
    df = df.with_columns(
        ((pl.col("_season_start") - pl.col("birth_date")).dt.total_days() / 365.25).alias("age_years"),
        pl.col("draft_pick").is_null().alias("is_undrafted"),
        pl.col("draft_pick").fill_null(UNDRAFTED_PICK_SENTINEL).alias("draft_pick"),
        pl.col("draft_round").fill_null(8).alias("draft_round"),  # 8 = beyond the 7-round draft
    )

    return df.select(
        [
            "player_id",
            "season",
            "position",
            "age_years",
            "years_exp",
            "draft_year",
            "draft_round",
            "draft_pick",
            "is_undrafted",
        ]
    )
