"""Red-zone / inside-10 opportunity share -- touchdown equity.

TD equity concentrates in short-yardage work, and docs/features.md §1d flags
this as real signal that is nonetheless low-n per game (0-2 touches in a
week), so every column here is trailing-window smoothed before use.

**This module previously computed the wrong quantity.** The old denominator
was the player's *own* touches -- `rush_rz_share` meant "what fraction of
this player's carries came in the red zone", which makes a third-string back
handed the ball once from the four-yard line the most valuable rusher in the
league at 1.000, while a workhorse with 18 carries including 3 inside the 20
scores 0.167. That is close to the inverse of what the feature is for.

The denominator is now the *team's* red-zone touches, so the columns mean
what their names claim: this player's share of his offense's scoring-range
work. Counts are emitted alongside shares, because share alone cannot
distinguish a large slice of an offense that never reaches the red zone from
a large slice of one that lives there.
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import ROLLING_WINDOWS, normalize_team
from nflforecast.features.utils import add_inclusive_rolling

RZ_YARDLINE = 20
I10_YARDLINE = 10


def _zone_shares(pbp: pl.DataFrame, filter_expr: pl.Expr, id_col: str, prefix: str) -> pl.DataFrame:
    """Player share of team touches inside the 20 and inside the 10, per week."""
    plays = pbp.filter(
        filter_expr & pl.col("yardline_100").is_not_null() & pl.col("posteam").is_not_null()
    ).with_columns(
        (pl.col("yardline_100") <= RZ_YARDLINE).alias("_rz"),
        (pl.col("yardline_100") <= I10_YARDLINE).alias("_i10"),
    )

    # Team denominator: all of the offense's red-zone touches of this type,
    # including those by players with no id (rare) -- the pool the player is
    # competing for a slice of.
    team = plays.group_by(["posteam", "season", "week"]).agg(
        pl.col("_rz").sum().alias("_team_rz"),
        pl.col("_i10").sum().alias("_team_i10"),
    )

    player = (
        plays.filter(pl.col(id_col).is_not_null())
        .group_by([id_col, "posteam", "season", "week"])
        .agg(
            pl.col("_rz").sum().alias(f"{prefix}_rz_n"),
            pl.col("_i10").sum().alias(f"{prefix}_i10_n"),
        )
        .rename({id_col: "player_id"})
    )

    joined = player.join(team, on=["posteam", "season", "week"], how="left")
    return joined.with_columns(
        # Guard the zero-denominator week: an offense that never crossed the 20
        # gives every player a null share, not a divide-by-zero.
        pl.when(pl.col("_team_rz") > 0)
        .then(pl.col(f"{prefix}_rz_n") / pl.col("_team_rz"))
        .alias(f"{prefix}_rz_share"),
        pl.when(pl.col("_team_i10") > 0)
        .then(pl.col(f"{prefix}_i10_n") / pl.col("_team_i10"))
        .alias(f"{prefix}_i10_share"),
    ).select(
        [
            "player_id",
            "season",
            "week",
            f"{prefix}_rz_n",
            f"{prefix}_i10_n",
            f"{prefix}_rz_share",
            f"{prefix}_i10_share",
        ]
    )


def build_redzone_features(pbp: pl.DataFrame) -> pl.DataFrame:
    reg = pbp.filter(pl.col("season_type") == "REG").with_columns(normalize_team("posteam"))

    rush_rz = _zone_shares(reg, pl.col("rush_attempt") == 1, "rusher_player_id", "rush")
    tgt_rz = _zone_shares(reg, pl.col("pass_attempt") == 1, "receiver_player_id", "tgt")

    df = rush_rz.join(tgt_rz, on=["player_id", "season", "week"], how="full", coalesce=True)

    # A player who ran but was not targeted has no target row and vice versa;
    # zero touches is a real observation here, not missing data.
    value_cols = [c for c in df.columns if c not in ("player_id", "season", "week")]
    df = df.with_columns([pl.col(c).fill_null(0) for c in value_cols])

    return add_inclusive_rolling(
        df, group_col="player_id", order_cols=["season", "week"],
        value_cols=value_cols, windows=list(ROLLING_WINDOWS),
    ).drop(value_cols)
