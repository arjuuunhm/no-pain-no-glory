"""Game-script usage splits, at player-week grain.

For each player-week, the share of that player's rush attempts / targets that
came while trailing by 8+, within one score, or leading by 8+
(`score_differential` is posteam-perspective in nflverse pbp).

This is the feature resources.md §6 and docs/features.md §9 call out as
separating top from median Big Data Bowl entries, and it is the dominant
driver of carry volume for a lead back specifically: a back whose carries are
concentrated in leading-big scripts is being fed by game states his offense
may not reproduce, and his volume is far more fragile than the raw carry
count suggests.

Note the denominator here is intentionally the *player's own* touches -- this
describes how a player was used, not how much of the offense he commanded
(that is opportunity.py's job, and red-zone share's). Because the game state
is partly a consequence of the outcome being predicted, these are rolled
forward and bound by attach_asof like every other player-grain block.
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import ROLLING_WINDOWS
from nflforecast.features.utils import add_inclusive_rolling

BLOWOUT_MARGIN = 8


def _bucket_shares(pbp: pl.DataFrame, filter_expr: pl.Expr, id_col: str, prefix: str) -> pl.DataFrame:
    plays = pbp.filter(
        filter_expr & pl.col(id_col).is_not_null() & pl.col("score_differential").is_not_null()
    )
    agg = (
        plays.group_by([id_col, "season", "week"])
        .agg(
            pl.len().alias("_total"),
            (pl.col("score_differential") <= -BLOWOUT_MARGIN).sum().alias("_trailing_big"),
            (pl.col("score_differential") >= BLOWOUT_MARGIN).sum().alias("_leading_big"),
            pl.col("score_differential").mean().alias(f"{prefix}_mean_score_diff"),
        )
        .rename({id_col: "player_id"})
    )
    return agg.with_columns(
        (pl.col("_trailing_big") / pl.col("_total")).alias(f"{prefix}_share_trailing_big"),
        (pl.col("_leading_big") / pl.col("_total")).alias(f"{prefix}_share_leading_big"),
        (
            (pl.col("_total") - pl.col("_trailing_big") - pl.col("_leading_big")) / pl.col("_total")
        ).alias(f"{prefix}_share_close"),
    ).drop(["_total", "_trailing_big", "_leading_big"])


def build_game_script_features(pbp: pl.DataFrame) -> pl.DataFrame:
    reg = pbp.filter(pl.col("season_type") == "REG")

    rush = _bucket_shares(reg, pl.col("rush_attempt") == 1, "rusher_player_id", "rush")
    tgt = _bucket_shares(reg, pl.col("pass_attempt") == 1, "receiver_player_id", "tgt")

    df = rush.join(tgt, on=["player_id", "season", "week"], how="full", coalesce=True)

    value_cols = [c for c in df.columns if c not in ("player_id", "season", "week")]
    return add_inclusive_rolling(
        df, group_col="player_id", order_cols=["season", "week"],
        value_cols=value_cols, windows=list(ROLLING_WINDOWS),
    ).drop(value_cols)
